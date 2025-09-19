clear;clc;close all;

% Load data
smvi = readmatrix("E:\POSTDOC\FD_predcition\processed_data\weekly\era5_week_SMVI_2000_2024.txt");
latlon = readmatrix("E:\POSTDOC\FD_predcition\processed_data\latlon.txt");

% Filter for 2012, months 3-5 (March-May)
idx = (smvi(:,1)==2012) & (smvi(:,2)>=3) & (smvi(:,2)<=5);
data_idx = smvi(idx, 5:end)';

% Load CONUS shapefile
conus_shp = shaperead("E:\PhD\4_ECA_SM_DR_T\CONUS_Shapefile\CONUS2\conus2.shp");

% Define drought colormap (US Drought Monitor style) - the good one
drought_colors = [
    1.0, 1.0, 1.0;        % White - No drought
    1.0, 1.0, 0.0;        % Yellow - Abnormally dry
    0.988, 0.827, 0.498;  % Light orange - Moderate
    1.0, 0.667, 0.0;      % Orange - Severe
    0.902, 0.0, 0.0;      % Red - Extreme
    0.451, 0.0, 0.0       % Dark red - Exceptional
];

% Define contour levels for SMVI range (0 to 3.0)
contour_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

% Pre-process data once (outside loop for efficiency)
lon = latlon(:,1);
lat = latlon(:,2);

% Create grid once
lon_range = [min(lon), max(lon)];
lat_range = [min(lat), max(lat)];
[lon_grid, lat_grid] = meshgrid(linspace(lon_range(1), lon_range(2), 200), ...
                                linspace(lat_range(1), lat_range(2), 150));

% Create CONUS mask ONCE (outside loop)
disp('Creating CONUS mask (one time only)...');
conus_mask = false(size(lon_grid));
for i = 1:length(conus_shp)
    if ~isempty(conus_shp(i).X) && ~isempty(conus_shp(i).Y)
        % Remove NaN values from shapefile coordinates
        x_shape = conus_shp(i).X;
        y_shape = conus_shp(i).Y;
        valid_shape = ~isnan(x_shape) & ~isnan(y_shape);
        x_shape = x_shape(valid_shape);
        y_shape = y_shape(valid_shape);
        
        if length(x_shape) > 2 % Need at least 3 points for polygon
            % Create mask using inpolygon
            mask_part = inpolygon(lon_grid, lat_grid, x_shape, y_shape);
            conus_mask = conus_mask | mask_part;
        end
    end
end
disp('Mask created! Now plotting...');

% Create large figure for 3x4 subplot
figure('Position', [50, 50, 2000, 1500]);
set(gcf, 'Color', 'white');

% Loop for 12 weeks (much faster now!)
for wk = 1:12
    % Get values for current week
    values = data_idx(:,wk);
    
    % Remove NaN values and invalid data
    valid_idx = ~isnan(values) & ~isinf(values);
    lon_clean = lon(valid_idx);
    lat_clean = lat(valid_idx);
    values_clean = values(valid_idx);
    
    % Clip values to reasonable SMVI range
    values_clean(values_clean < 0) = 0;
    values_clean(values_clean > 3) = 3;
    
    % Interpolate data to grid (linear method)
    values_grid = griddata(lon_clean, lat_clean, values_clean, lon_grid, lat_grid, 'linear');
    
    % Apply the pre-computed mask
    values_grid(~conus_mask) = NaN;
    
    % Create subplot
    subplot(3, 4, wk);
    
    % Create filled contour plot with proper levels
    [C, h] = contourf(lon_grid, lat_grid, values_grid, contour_levels, 'LineStyle', 'none');
    
    % Apply colormap
    colormap(drought_colors);
    
    % Plot CONUS boundary on top
    hold on;
    for i = 1:length(conus_shp)
        if ~isempty(conus_shp(i).X)
            plot(conus_shp(i).X, conus_shp(i).Y, 'k-', 'LineWidth', 1.2);
        end
    end
    
    % Set axis limits to focus on CONUS
    xlim([lon_range(1)-0.5, lon_range(2)+0.5]);
    ylim([lat_range(1)-0.5, lat_range(2)+0.5]);
    
    % Remove axis labels, ticks, and box
    set(gca, 'XTick', [], 'YTick', []);
    set(gca, 'Box', 'off');
    axis off;
    
    % Add subplot title with large font
    subplot_letters = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', ...
                      '(g)', '(h)', '(i)', '(j)', '(k)', '(l)'};
    title([subplot_letters{wk} ' Week-' num2str(wk)], ...
          'FontSize', 16, 'FontWeight', 'bold');
    
    % Set consistent color limits for all subplots (SMVI range 0 to 3.0)
    caxis([0, 3.0]);
    
    % Make subplot area tight
    set(gca, 'Position', get(gca, 'Position') .* [1 1 1.05 1.05]);
end

% Add single colorbar for the entire figure (positioned at the right)
c = colorbar('Position', [0.92, 0.15, 0.025, 0.7]);

% Format colorbar with large fonts
c.FontSize = 18;
c.FontWeight = 'bold';
c.LineWidth = 2;
c.Label.String = 'SMVI';
c.Label.FontSize = 20;
c.Label.FontWeight = 'bold';

% Set colorbar ticks for SMVI range (0 to 3.0)
c.Ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
c.TickLabels = {'0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'};

% Add main title
% sgtitle('2012 Spring SMVI Evolution (Weekly)', 'FontSize', 22, 'FontWeight', 'bold');

% Save the figure
print('2012-fd_smvi.png', '-dpng', '-r300');

% Display completion message
disp('Clean SMVI 12-week subplot created successfully!');
disp('File saved: 2012-fd_smvi.png');
disp('Issues fixed: ');
disp('- Masked values outside CONUS boundary');
disp('- Used linear interpolation to avoid color artifacts');
disp('- Clipped extreme values to 0-3 range');
disp('- Improved boundary masking with inpolygon');