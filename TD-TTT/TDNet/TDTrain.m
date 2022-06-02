function TDTrain(S, r)
% Train the TD system
% S - The states' history for a given trial
% r - the reward given for each state in the given trial

% Load the network's data if needed
global Net;
LoadNet();

% TODO: Implement TD(0). It is highly recommended to read the documentation
% of TDEvaluate first.

eta = 1e-4;
gama = 0.9;
n_turns = length(r);

W = Net.W{1};
w_delta = zeros(n_turns, length(W));

for t = 1:n_turns
    V(t) = 0;
    for tau = 1:t - 1
        V(t) = V(t) + TDEvaluate(S(:, t-tau));
        %%V(t) = V(t) + W * [S(:, t-tau); 1];
    end
    
    delta(t) = sum(r(t:end)) - V(t) * gama ^ t;
    
    for tau = 1:t - 1
        w_delta(tau, :) = eta * delta(t) * [S(:, t-tau); 1]';
    end
    w_delta = sum(w_delta, 1);
    W = W + w_delta;
end
Net.W{1} = W;

end

