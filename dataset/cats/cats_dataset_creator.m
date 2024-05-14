clear all, close all, clc;

Folder = "OUTDOOR";

LWIPoints = [456 59; 568 211; 84 202; 256 221];
RGBPoints = [505 16; 640 196; 63 183; 272 206];
HLEFT = fitgeotrans(LWIPoints,RGBPoints,'projective');

LWIPoints = [481 29; 584 181; 102 174; 267 192];
RGBPoints = [495 12; 620 191; 49 181; 249 203];
HRIGHT = fitgeotrans(LWIPoints,RGBPoints,'projective');

% make the output folder
OutputFolder = 'cats_train';
LWIOutputFolder = fullfile(OutputFolder, 'ir');
RGBOutputFolder = fullfile(OutputFolder, 'rgb');

mkdir(OutputFolder);
mkdir(LWIOutputFolder);
mkdir(RGBOutputFolder);

% helper functions
RemoveDotFolders = @(d) (d(3:end));
GetSubFolders = @(p) (RemoveDotFolders(dir(p)));

% get the all subfolder
CatsClasses = GetSubFolders(Folder);
for c = 1:length(CatsClasses)
    ClassPath = fullfile(CatsClasses(c).folder, CatsClasses(c).name);
    CatsClassScenes = GetSubFolders(ClassPath);
    for s = 1:length(CatsClassScenes)
        RawImagePath = fullfile(ClassPath, CatsClassScenes(s).name, 'rawImages');

        % get left and right images
        try
            LC = imread(fullfile(RawImagePath, 'left_color_default.png'));
            LT = imread(fullfile(RawImagePath, 'left_thermal_default.png'));
            RC = imread(fullfile(RawImagePath, 'right_color_default.png'));
            RT = imread(fullfile(RawImagePath, 'right_thermal_default.png'));
            
            % resize color images
            LC = imresize(LC, 0.5);
            RC = imresize(RC, 0.5);
            
            % equalize histogram on 16bit images
            LT = adapthisteq(LT, "ClipLimit", 0.01);
            RT = adapthisteq(RT, "ClipLimit", 0.01);
%             LT = histeq(LT);
%             RT = histeq(RT);
            
%             LTa = LT;
%             RTa = RT;
            [LTa, ref] = imwarp(LT,HLEFT,'OutputView',imref2d(size(LT)));
            [RTa, ref] = imwarp(RT,HRIGHT,'OutputView',imref2d(size(RT)));

            LT = uint8(double(LTa) ./ 256);
            RT = uint8(double(RTa) ./ 256);
            
            imwrite(LC, fullfile(RGBOutputFolder, lower(sprintf('%s_%s_%02dlv.png', Folder, CatsClasses(c).name, s))));
            imwrite(LT, fullfile(LWIOutputFolder, lower(sprintf('%s_%s_%02dlt.png', Folder, CatsClasses(c).name, s))));
            imwrite(RC, fullfile(RGBOutputFolder, lower(sprintf('%s_%s_%02drv.png', Folder, CatsClasses(c).name, s))));
            imwrite(RT, fullfile(LWIOutputFolder, lower(sprintf('%s_%s_%02drt.png', Folder, CatsClasses(c).name, s))));
        catch
            fprintf(lower(sprintf('%s_%s_%02dl not exist\n', Folder, CatsClasses(c).name, s)));
        end
    end
end