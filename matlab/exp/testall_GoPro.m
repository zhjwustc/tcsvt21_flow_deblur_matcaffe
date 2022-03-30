clear

addpath('../');

caffe.reset_all()

datapath = '/pathtodata/';

folderlist = {'GOPR0881_11_01','GOPR0871_11_00','GOPR0869_11_00','GOPR0868_11_00','GOPR0862_11_00','GOPR0854_11_00','GOPR0410_11_00','GOPR0396_11_00','GOPR0385_11_01','GOPR0384_11_05','GOPR0384_11_00'};

model_path = 'model/';

Solver = modelconfig_test(model_path);

mkdir('results');

for i=1:length(folderlist)
    imlist = dir(fullfile(datapath, folderlist{i}, 'blur', '*.png'));
    mkdir(fullfile('results', folderlist{i}))
    for j=1:length(imlist)-1
        src = im2single(imread(fullfile(datapath, folderlist{i}, 'blur', imlist(j).name)));
        dst = im2single(imread(fullfile(datapath, folderlist{i}, 'blur', imlist(j+1).name)));
        
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

