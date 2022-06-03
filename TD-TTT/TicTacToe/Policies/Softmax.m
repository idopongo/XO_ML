function num = Softmax(Grades)
%EPSGREEDY Softmax policy
% Grades    - The critic grades for each possible action
% num       - The chosen action's index


result = exp(Grades)./sum(exp(Grades));
num = find(rand<cumsum(result),1);

end
