%% =======================
%  Global parameters
%% =======================
source_frequencies = 500000;     % [Hz]
source_roc      = 0.0632;        % [m]
source_diameter = 0.064;         % [m]
source_mag      = 60000;         % [Pa]

cmin = 1500;
cmax = 3200;
dx = 4.8e-04;
Nx = 512; Ny = 512;
cfl = 0.2;
dt = cfl * dx/cmax;
t_end = 0.9e-4;
record_periods = 1;
Nt = round(t_end / dt);
PPP = 1/(source_frequencies*dt);

bli_tolerance   = 0.1;
upsampling_rate = 8;

% PixelSpacing check
targetPS = [0.4883; 0.4883];
psTol = 1e-4;

% Slice selection rules
% n > 35 -> middle 10
% n > 30 -> middle 8
% n <= 30 -> skip

% Postprocess / SDF
minArea     = 10;
conn        = 8;
linkRadius  = 2;
maxHoleArea = 50;   % 只补小洞（避免圆环中心被填）；0=不补洞
sdfClip     = 50;
sdfNorm     = true;

%% =======================
%  Paths (EDIT HERE)
%% =======================
rootDir = "F:\HeadCT\ctsinogram\head_ct_dataset_anon";

% 指定输出根目录（你要改到你希望的路径）
outRoot = "F:\HeadCT\export_labels";

outPressure = fullfile(outRoot, "pressure");
outDB       = fullfile(outRoot, "dB");
outSDF      = fullfile(outRoot, "sdf");

if ~exist(outPressure, "dir"), mkdir(outPressure); end
if ~exist(outDB, "dir"),       mkdir(outDB); end
if ~exist(outSDF, "dir"),      mkdir(outSDF); end

%% =======================
%  Scan all batches / series
%% =======================
batchDirs = dir(fullfile(rootDir, "batch_*"));
batchDirs = batchDirs([batchDirs.isdir]);

fprintf("Found %d batches.\n", numel(batchDirs));

for ib = 1:numel(batchDirs)
    batchPath = fullfile(batchDirs(ib).folder, batchDirs(ib).name);
    seriesDirs = dir(fullfile(batchPath, "series_*"));
    seriesDirs = seriesDirs([seriesDirs.isdir]);

    fprintf("\n== %s : %d series ==\n", batchDirs(ib).name, numel(seriesDirs));

    for is = 1:numel(seriesDirs)
        seriesName = seriesDirs(is).name;
        seriesPath = fullfile(seriesDirs(is).folder, seriesName);
        reconPath  = fullfile(seriesPath, "reconstructed_image");

        if ~exist(reconPath, "dir")
            continue;
        end

        % Get all dcm slices in reconstructed_image
        dcmFiles = dir(fullfile(reconPath, "image_*.dcm"));
        if isempty(dcmFiles)
            continue;
        end

        % Sort by numeric index in filename: image_XX.dcm
        idx = zeros(numel(dcmFiles),1);
        for k = 1:numel(dcmFiles)
            tok = regexp(dcmFiles(k).name, "image_(\d+)\.dcm", "tokens", "once");
            idx(k) = str2double(tok{1});
        end
        [~, order] = sort(idx);
        dcmFiles = dcmFiles(order);

        % Check PixelSpacing in first dicom
        firstFn = fullfile(dcmFiles(1).folder, dcmFiles(1).name);
        try
            info = dicominfo(firstFn);
            if ~isfield(info, "PixelSpacing") || isempty(info.PixelSpacing)
                fprintf("Skip %s (no PixelSpacing)\n", seriesName);
                continue;
            end
            ps = double(info.PixelSpacing(:));
            if numel(ps) ~= 2 || any(abs(ps - targetPS) > psTol)
                fprintf("Skip %s (PixelSpacing=%s)\n", seriesName, mat2str(ps));
                continue;
            end
        catch ME
            fprintf("Skip %s (dicominfo error: %s)\n", seriesName, ME.message);
            continue;
        end

        nSlices = numel(dcmFiles);
        if nSlices <= 30
            fprintf("Skip %s (nSlices=%d <= 30)\n", seriesName, nSlices);
            continue;
        elseif nSlices > 35
            kSlices = 10;
        else
            kSlices = 8;
        end

        startIdx = floor((nSlices - kSlices)/2) + 1;
        selIdx = startIdx : (startIdx + kSlices - 1);

        fprintf("Process %s: nSlices=%d, take %d slices [%d..%d]\n", ...
            seriesName, nSlices, kSlices, selIdx(1), selIdx(end));

        %% =======================
        %  Process selected slices
        %% =======================
        for ii = selIdx
            fn = fullfile(dcmFiles(ii).folder, dcmFiles(ii).name);

            try
                % ---- Read CT slice
                P = dicomread(fn);
                P(P < 1000) = 1000;
                P = rot90(P, 3);

                % ---- Create grid/time
                kgrid = kWaveGrid(Nx, dx, Ny, dx);
                kgrid.setTime(Nt, dt);

                % ---- Source signal
                source_sig = createCWSignals(kgrid.t_array, source_frequencies, source_mag, 0);

                % ---- Setup array
                karray = kWaveArray('BLITolerance', bli_tolerance, 'UpsamplingRate', upsampling_rate);

                row_c = round(Ny/2);
                col_skull = find(P(row_c,:) > 1000, 1, "first");
                if isempty(col_skull) || col_skull <= 25
                    fprintf("  Skip slice %s (invalid col_skull)\n", dcmFiles(ii).name);
                    continue;
                end
                col_src = col_skull - 20;

                x0 = (row_c - round(Nx/2)) * dx;
                y0 = (col_src - (Ny/2)) * dx;

                arc_pos  = [x0, y0];
                arc_rad  = source_roc;
                arc_diam = source_diameter;
                focus_pos = [x0, y0 + arc_rad];

                karray = kWaveArray('BLITolerance', bli_tolerance, 'UpsamplingRate', upsampling_rate);
                karray.addArcElement(arc_pos, arc_rad, arc_diam, focus_pos);

                source.p_mask = karray.getArrayBinaryMask(kgrid);
                source.p      = karray.getDistributedSourceSignal(kgrid, source_sig);

                % ---- Medium
                rho = hounsfield2density(single(P));
                v = 0.70 .* rho + 1014;

                skull_mask = P > 1300;
                alpha_coef = 13.3 * skull_mask;

                medium.sound_speed  = v;
                medium.density      = rho;
                medium.alpha_coeff  = alpha_coef;
                medium.alpha_power  = 2;

                % ---- Sensor
                sensor.mask = zeros(Nx, Ny);
                sensor.mask(:, col_src+2:end) = 1;

                sensor.record = {'p'};
                sensor.record_start_index = kgrid.Nt - record_periods * PPP + 1;

                input_args = {
                    'PMLSize', [16, 16], ...
                    'PMLAlpha', 1.5, ...
                    'PMLInside', false, ...
                    'PlotPML', false, ...
                    'DisplayMask', 'off'
                };

                % ---- Run
                sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
                    input_args{:}, ...
                    'DataCast', 'gpuArray-single', ...
                    'PlotLayout', false, ...
                    'PlotScale', [-1, 1] * source_mag);

                % ---- Amp
                sensor_data_reshaped_p = reshape(sensor_data.p, Nx, Ny - col_src - 1, []);
                amp = extractAmpPhase(sensor_data_reshaped_p, 1 / kgrid.dt, source_frequencies, ...
                    'Dim', 3, 'Window', 'Rectangular', 'FFTPadding', 1);
                amp = reshape(amp, Nx, []);
                amp = gather(amp);

                % ---- dB zones
                [dB3, dB6, dB9] = makeDbZones(amp);

                % ---- postprocess masks (output already 512x512)
                dB3_prc = fieldPost(dB3, skull_mask, minArea, conn, linkRadius, maxHoleArea);
                dB6_prc = fieldPost(dB6, skull_mask, minArea, conn, linkRadius, maxHoleArea);
                dB9_prc = fieldPost(dB9, skull_mask, minArea, conn, linkRadius, maxHoleArea);

                % ---- SDF from processed masks
                dB3_SDF = mask2sdf(dB3_prc, sdfClip, sdfNorm);
                dB6_SDF = mask2sdf(dB6_prc, sdfClip, sdfNorm);
                dB9_SDF = mask2sdf(dB9_prc, sdfClip, sdfNorm);

                % ---- Save (use unique name)
                baseName = sprintf("%s_%s_%s", batchDirs(ib).name, seriesName, erase(dcmFiles(ii).name, ".dcm"));
                save(fullfile(outPressure, baseName + ".mat"), "amp", "-v7.3");
                save(fullfile(outDB,       baseName + ".mat"), "dB3", "dB6", "dB9", "-v7.3");
                save(fullfile(outSDF,      baseName + ".mat"), "dB3_SDF", "dB6_SDF", "dB9_SDF", "-v7.3");

                fprintf("  Saved: %s\n", baseName);

            catch ME
                fprintf("  ERROR %s: %s\n", dcmFiles(ii).name, ME.message);
                continue;
            end
        end
    end
end

fprintf("\nDone.\n");
