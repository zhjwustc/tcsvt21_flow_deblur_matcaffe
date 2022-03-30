warning off

% delete(gcp)
% clear all

% parpool('local', 10);

addpath('../');
caffe.reset_all()

model_path = 'model/';

Solver = modelconfig(model_path);
% state_path = Solver.state_path;

Solver = dataconfig(Solver);

if isfield(Solver, 'iter')
    begin = Solver.iter+1;
else
    begin = 1;
    Solver = usePreTrainedModel(Solver);
%     Solver = usePreTrainedModel2(Solver);
end

for iter = begin:Solver.max_iter
    Solver.iter = iter;
    Solver.Solver_.set_iter(iter);
    
    [batch, gt] = Gen_training( Solver);                                                                
    batchc = {single(batch)};
    activec = Solver.Solver_.net.forward(batchc);
    
%     figure(1); imshow([gt(:,:,:,1), batch(:,:,1:3,1),imresize(activec{4}(:,:,:,1),2,'nearest'),activec{1}(:,:,:,1)])

%     drawnow;

    [deltac{1}, loss1_deblur] = L2Loss(activec{1}, gt, 'train');
    [deltac{2}, loss1_flow] = L2Loss(activec{2}, imresize(batch(:,:,1:3,:),1/4), 'train');
    [deltac{3}, loss2_flow] = L2Loss(activec{3}, imresize(batch(:,:,1:3,:),1/8), 'train');
    [deltac{4}, loss3_flow] = L2Loss(activec{4}, imresize(batch(:,:,1:3,:),1/16), 'train');
    [deltac{5}, loss4_flow] = L2Loss(activec{5}, imresize(batch(:,:,1:3,:),1/32), 'train');
    [deltac{6}, loss5_flow] = L2Loss(activec{6}, imresize(batch(:,:,1:3,:),1/64), 'train');
    
%     deltac{4} = deltac{4}*4;
%     deltac{5} = deltac{5}*4;
%     deltac{6} = deltac{6}*4;
%     deltac{7} = deltac{7}*4;
    
    Solver.loss1_deblur(iter) = loss1_deblur(1);
    Solver.loss1_flow(iter) = loss1_flow(1);
    Solver.loss2_flow(iter) = loss2_flow(1);
    Solver.loss3_flow(iter) = loss3_flow(1);
    Solver.loss4_flow(iter) = loss4_flow(1);
    Solver.loss5_flow(iter) = loss5_flow(1);
    
    if ~isnan(Solver.loss1_deblur(iter))
        Solver.Solver_.net.backward(deltac);
        Solver.Solver_.update();
    else
        error('Model NAN.')
    end
    
    % vis
    if ~mod(iter,10)
        fprintf('========Processed iter %.6d, ',iter);
        fprintf('loss deblur: %d=======', mean(Solver.loss1_deblur(iter-9:iter)));
        fprintf('loss flow: %d=======', mean(Solver.loss1_flow(iter-9:iter)));
        fprintf('\n');
    end    
    
    if ~mod(iter,1000)
        
        imwrite([gt(:,:,:,1),batch(:,:,1:3,1),imresize(activec{2}(:,:,:,1),4,'nearest'),activec{1}(:,:,:,1)], fullfile('out', [num2str(iter), '.png']))
        
        Solver.Solver_.save();
        % save mat
        delete(['./model/LRNN_iter_' num2str(iter-2000) '.caffemodel'])
        delete(['./model/LRNN_iter_' num2str(iter-2000) '.solverstate'])
        save(Solver.matfile, 'Solver');
    end
    
    
    
    
    
end
