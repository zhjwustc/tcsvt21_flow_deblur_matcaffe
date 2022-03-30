folderlist = {'720p_240fps_2','IMG_0003','IMG_0021','IMG_0030','IMG_0031','IMG_0032','IMG_0033','IMG_0037','IMG_0039','IMG_0049'};

gtfolder = '/data/deblur_test_set/deepvideo';
resfolder = '/data/deblur/res';

psnrres = zeros(10,99);

for i=1:length(folderlist)
    reslist = dir(fullfile(resfolder, folderlist{i}, '*.png'));
    gtlist = dir(fullfile(gtfolder, folderlist{i}, 'GT', '*.jpg'));
    for j=1:length(reslist)
        res = imread(fullfile(resfolder, folderlist{i}, reslist(j).name));
        gt = imread(fullfile(gtfolder, folderlist{i}, 'GT', gtlist(j).name));
        res = im2double(res);
        gt = im2double(gt);
        psnrres(i,j) = psnr(res, gt);
    end
end

mean(psnrres(:))
save('psnrres','psnrres')
