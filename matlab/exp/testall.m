clear

addpath('../');

caffe.reset_all()

datapath = './';

folderlist = {'720p_240fps_2','IMG_0003','IMG_0021','IMG_0030','IMG_0031','IMG_0032','IMG_0033','IMG_0037','IMG_0039','IMG_0049'};

model_path = 'model/';

Solver = modelconfig_test(model_path);

mkdir('results');

for i=1:length(folderlist)
    imlist = dir(fullfile(datapath, folderlist{i}, 'input', '*.jpg'));
    mkdir(fullfile('results', folderlist{i}))
   for j=1:length(imlist)-1
        src = im2single(imread(fullfile(datapath, folderlist{i}, 'input', imlist(j).name)));
        dst = im2single(imread(fullfile(datapath, folderlist{i}, 'input', imlist(j+1).name)));
        
        [h,w,~] = size(src);
        if(h>w)
            src = permute(src,[2,1,3]);
            dst = permute(dst,[2,1,3]);
        end

        src = [src; zeros(48,1280,3)];
        dst = [dst; zeros(48,1280,3)];

        batch = cat(3, src, dst);
        batchc = {single(batch)};

        tic
        activec = Solver.Solver_.net.forward(batchc);
        toc

        active = activec{1};
        active = active(1:720,:,:);
        

        if(h>w)
            active = permute(active,[2,1,3]);
        end
        
        imwrite([active], fullfile('results', folderlist{i}, [imlist(j).name(1:end-4), '.png']))
    end
end

