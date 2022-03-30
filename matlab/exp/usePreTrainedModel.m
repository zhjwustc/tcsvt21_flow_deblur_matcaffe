function Solver = usePreTrainedModel(Solver)
%%%%%% use it before 1st iteration in train
w1 = Solver.Solver_.net.get_weights();
w2 = load('./model/xxxx.mat'); w2 = w2.weights;

w1 = w2;

% for i=1:45
%     w1(i).weights = w2(i).weights;
% end
% 
% for i=47:47
%     w1(i+1).weights = w2(i).weights;
% end
% 
% for i=48:48
%     w1(i+2).weights = w2(i).weights;
% end
% 
% for i=49:49
%     w1(i+3).weights = w2(i).weights;
% end

Solver.Solver_.net.set_weights(w1);
