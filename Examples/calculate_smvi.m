%% ------------------ CONFIG ------------------
clear;clc;close; 
in_fp  = "E:\POSTDOC\FD_predcition\processed_data\weekly\era5_week_soil_moisture_conus_2000_2024.txt";
out_fp = "E:\POSTDOC\FD_predcition\processed_data\weekly\era5_week_SMVI_2000_2024.txt";

lagWeeks = 1;    % change over 1 week (weekly data)
winWeeks = 4;    % rolling window for realized volatility (e.g., 4 = ~1 month)

%% ------------------ LOAD --------------------
M = readmatrix(in_fp);             % [T x (4+Ngrid)]
year  = M(:,1);
month = M(:,2);
woy   = M(:,3);                    % week-of-year (1..52)
day   = M(:,4);
S     = M(:,5:end);                % soil moisture (weekly), T x Ngrid

[T, N] = size(S);

%% ------------------ WEEKLY CLIM (MEAN/STD) --
% Compute weekly climatology per grid using explicit loops
mu  = nan(52, N);                  % weekly mean
sg  = nan(52, N);                  % weekly std
for w = 1:52
    idx = (woy == w);
    if any(idx)
        X = S(idx, :);
        mu(w, :) = mean(X, 1, 'omitnan');
        sg(w, :) = std (X, 0, 'omitnan');
    end
end
% avoid division by zero
sg(sg < 1e-8) = 1e-8;

% Map weekly climatology to the full time vector
MU = nan(T, N);  SG = nan(T, N);
for i = 1:T
    MU(i,:) = mu(woy(i), :);
    SG(i,:) = sg(woy(i), :);
end

%% ------------------ STANDARDIZED ANOMALY ----
Z = (S - MU) ./ SG;                 % z-score per week/grid

%% ------------------ LAGGED CHANGE dZ --------
dZ = nan(T, N);
dZ(2:end, :) = Z(2:end, :) - Z(1:end-1, :);

% Optional: guard against accidental jumps (e.g., missing rows).
% If you know the file is strictly weekly sequential, you can skip this.
validStep = false(T,1);
validStep(2:end) = ( (year(2:end) == year(1:end-1)   & woy(2:end) == woy(1:end-1)+1) | ...
                     (year(2:end) == year(1:end-1)+1 & woy(2:end) == 1 & woy(1:end-1) >= 52) );
dZ(~validStep, :) = NaN;

%% ------------------ REALIZED VOLATILITY (RV) -
% sqrt of rolling mean of squared weekly changes over winWeeks
RV = sqrt( movmean(dZ.^2, [winWeeks-1 0], 1, 'omitnan') );

%% ------------------ WEEKLY RV CLIMATOLOGY ----
rv_mu = nan(52, N);
for w = 1:52
    idx = (woy == w);
    if any(idx)
        rv_mu(w, :) = mean(RV(idx, :), 1, 'omitnan');
    end
end
rv_mu(rv_mu < 1e-8) = 1e-8;

% map weekly RV baseline to time
RVb = nan(T, N);
for i = 1:T
    RVb(i,:) = rv_mu(woy(i), :);
end

%% ------------------ SMVI ---------------------
SMVI = RV ./ RVb;                   % unitless; >1 = unusually volatile

%% ------------------ SAVE ---------------------
OUT = [year month woy day SMVI];
writematrix(OUT, out_fp);

% (Optional) quick sanity print
fprintf('SMVI written: %s\n', out_fp);
fprintf('Example stats across all grids: median=%.2f, 90th pct=%.2f\n', ...
        median(SMVI(:),'omitnan'), prctile(SMVI(:),90));
