addpath('../');
model_path = 'model/';
Solver = modelconfig(model_path);
weights=Solver.Solver_.net.get_weights();
save('weights','weights') 

% clear all;
% 
% addpath('../');
% caffe.reset_all()
% 
% 
% % Solver.Solver_ = caffe.Solver()
% 
% solver_file = './model/solver.prototxt';
% Solver = SolverParser(solver_file);
% Solver.Solver_ = caffe.Solver(solver_file);
% Solver.modelfile = './model/FlowNet2-s_weights.caffemodel';
% Solver.Solver_.net.copy_from(Solver.modelfile);
% 
% weights=Solver.Solver_.net.get_weights();
% 
% save('weights','weights') 