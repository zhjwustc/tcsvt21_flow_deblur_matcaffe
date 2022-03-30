function Solver = dataconfig(Solver)
Solver.patchsize = 256;
Solver.batchsize = 10;
%%% not all the data are used for training

Solver.datapath = '/pathtodata/'; 
Solver.srcfolder = 'SrcInput';
Solver.dstfolder = 'DstInput';
Solver.cleanfolder = 'SrcGt';
Solver.subfolderpath = dir(fullfile(Solver.datapath, Solver.srcfolder));
Solver.subfolderpath(1:2) = [];
if exist(fullfile(Solver.datapath, 'data.mat'),'file')
    load(fullfile(Solver.datapath, 'data.mat'));
else
    testlst = {};
    trainlst = {};
    
    count = 1;
    for subfolderid = 1:length(Solver.subfolderpath)
        dir_list = dir(fullfile(Solver.datapath,Solver.srcfolder,Solver.subfolderpath(subfolderid).name,'*.png'));
        num_jpg = length(dir_list)
        for id = 1:num_jpg
            trainlst{count}.src = fullfile(Solver.datapath,Solver.srcfolder,Solver.subfolderpath(subfolderid).name,dir_list(id).name);
            trainlst{count}.dst = fullfile(Solver.datapath,Solver.dstfolder,Solver.subfolderpath(subfolderid).name,dir_list(id).name);
            trainlst{count}.clean = fullfile(Solver.datapath,Solver.cleanfolder,Solver.subfolderpath(subfolderid).name,dir_list(id).name);
            count = count+1;
        end
    end
  

    data.trainlst = trainlst;
    data.train_num = length(trainlst);
    data.testlst = testlst;
    data.test_num = length(testlst);   
   
   
    fprintf('saving data structure ...\n');
    save(fullfile(Solver.datapath, 'data.mat'), 'data');
end
Solver.data = data;
fprintf('Done with data config, obtain %d traning images.\n',Solver.data.train_num);
end


