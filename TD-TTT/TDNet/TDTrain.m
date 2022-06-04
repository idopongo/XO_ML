function TDTrain(S, r, game_num)
% Train the TD system
% S - The states' history for a given trial
% r - the reward given for each state in the given trial

% Load the network's data if needed
global Net;
LoadNet();

% TODO: Implement TD(0). It is highly recommended to read the documentation
% of TDEvaluate first.

eta = 0.02 * (1 / (1+game_num*1e-4));
gama = 0.9;
n_turns = length(r);

for t = 1:n_turns
    [V_curr, grad] = TDEvaluate(S(:,t));
    
    if t < n_turns
        delta = r(t) + gama*TDEvaluate(S(:,t+1)) - V_curr;
    else
        delta = r(t) - V_curr;
    end
    
    for l = 1:length(Net.W)
        w_delta = eta * delta * reshape(grad{l}, size(Net.W{l}));
        Net.W{l} = Net.W{l} + w_delta;
    end
end

end

