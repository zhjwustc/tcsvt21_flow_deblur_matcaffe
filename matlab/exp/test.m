clear

addpath('../');

caffe.reset_all()

datapath = './';
src = im2single(imread(fullfile(datapath, '00008.jpg')));
dst = im2single(imread(fullfile(datapath, '00009.jpg')));

src = [src; zeros(48,1280,3)];
dst = [dst; zeros(48,1280,3)];

batch = cat(3, src, dst);
batchc = {single(batch)};

model_path = 'model/';

Solver = modelconfig_test(model_path);

% Solver.Solver_.net.blobs('data').reshape([size(src,1),size(src,2),3*2,1]);

tic
activec = Solver.Solver_.net.forward(batchc);
toc

active = activec{1};
active = active(1:720,:,:);

% imwrite([src], 'blur.png')
imwrite([active], 'deblur.png')

% flow1 = activec{1}(:,:,1);
% flow2 = activec{1}(:,:,2);
% flow1 = (flow1-min(flow1(:)))/(max(flow1(:))-min(flow1(:)));
% flow2 = (flow2-min(flow2(:)))/(max(flow2(:))-min(flow2(:)));
% imwrite(flow1, 'flow1.png')
% imwrite(flow2, 'flow2.png')

