% v1 support inputs with image gradients (x and y)
function [batch, gt] = Gen_training( Solver)

batch = single(zeros(Solver.patchsize,Solver.patchsize,6,Solver.batchsize));
gt = single(zeros(Solver.patchsize,Solver.patchsize,3,Solver.batchsize));
rng('shuffle');
idpool = randperm(Solver.data.train_num);
for count = 1:Solver.batchsize
    idx = idpool(count);
    src = im2single(imread(Solver.data.trainlst{idx}.src));
    dst = im2single(imread(Solver.data.trainlst{idx}.dst));
    clean = im2single(imread(Solver.data.trainlst{idx}.clean));
    
    batch(:,:,:,count) = cat(3, src, dst);
    gt(:,:,:,count) = clean;
end
end