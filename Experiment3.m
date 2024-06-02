
%% Experiment 3 - Performance on a two-armed bandit task with various levels of the optimism bias

% An Active Inference Model of the Optimism Bias 
% Elizabeth Fisher, Christopher Whyte, Jakob Hohwy
% 4/4/2024

% This is a two armed bandit task where one arm has a higher reward, but a
% higher loss and the other arm has a lower reward but a lower loss. The
% precision of the wins and losses in this task are conditioned on the
% optimism of the agent. The optimism is learnt through a simulation code
% and implemented this task

clear 
close all 
rng('default')

% Create struct to store resutls 
simulationResults = struct('loop', {}, 'variables', {}, 'winnings', {});

% Loop through the levels of optimism 
for loop = 1:9
rng('default')
  
ob = loop/10; % set the optimisim level

% Priors over the initial states
D{1} = [1 0 0]'; % location prior belief [ start, arm1, arm2]
D{2} = [ob (1-ob)]'; % Optimism [optimisim, pessimism]


% A{1} Location of the agent
   for l = 1:length(D{2}) % Optimism [optimisim, pessimism]
           A{1}(:,:,l) = eye(length(D{1}));
   end



% A{2} Reward Outcomes 
% The wins and losses in the generative process are the same regardless of
% the optimism of the agent 

   for l = 1:length(D{2}) % [optimisim, pessimism]
            A{2}(:,:,l) = [1  0    0; %  null
                           0  0.7  0; %  large win
                           0  0.3  0; %  large loss
                           0  0  0.6; %  small win
                           0  0  0.4];%  small loss
   end
  
     

% Seperate gen model from process
a = A;

% [start, arm1, arm2]
% Optimism
     a{2}(:,:,1) = [1    0    0;  % null
                    0   0.9  0.9;  % large win
                    0   0.1  0.1;  % large loss
                    0   0.9 0.9;  % small win
                    0   0.1 0.1]; % small loss
  

% [start, arm1, arm2]
% pessimism
      a{2}(:,:,2) = [1  0    0;  % null
                      0  0.1  0.1;  % large win
                      0  0.9  0.9;  % large loss
                      0  0.1  0.1;  % small win
                      0  0.9 0.9]; % small loss


a{1} = a{1}*100; 

% The B{1} matrix corresponds to the 3 policies. The columns are [start, arm1, arm2]
% You can move from any state to the chosen state


B{1}(:,:,1) = [1 1 1; % start
               0 0 0; % left
               0 0 0];%right 

B{1}(:,:,2) = [0 0 0;
               1 1 1;
               0 0 0];


B{1}(:,:,3) = [0 0 0;
               0 0 0;
               1 1 1];

% The agent cannot control the optimisim state, or the wins and losses state
% so there is only 1 'action', indicating that mood remains stable within a trial:

B{2} = [1 0; 
        0 1]; 

% The C matrix is the preference the agent has. The agent has a preference
% for reward in C{2} but no preference for location C{1}
C{1} = [0 0; 
        0 0;
        0 0];


C{2} = [0  0;  % null
        0  4;  % large win
        0 -4;  % large loss
        0  2;  % small win
        0 -2]; % small loss


% The V matrix shows the possible polices. 
% The agent can chose arm1 or arm2
V(:,:,1) = [1 2 3]; % Policy for chosing arm1, arm2
V(:,:,2) = [1 1 1]; % No policy for optimisim



% The number of time steps 
T = 2;

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % Policies.
mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % priors over initial states
mdp.a = a;
mdp.eta = 0.3;                % learning rate


mdp.Aname   = {'Win/Lose', 'Choice made'};
mdp.Bname   = {'Choice', 'Optimisim'};


label.factor{1}   = 'Location'; 
label.factor{2}   = 'Wins/Losses';   label.name{1}    = {'start','Arm1','Arm2'};   label.name{2}    = {'wins','losses'};
label.modality{2} = 'Location';    label.outcome{1} = {'Start','Arm1','Arm2'};
label.modality{1} = 'win/lose';  label.outcome{2} = {'null',' large win','large loss','small win', 'small loss'};
label.action{1} = {'start','Arm1','Arm2'};
mdp.label = label;

mdp = spm_MDP_check(mdp);

% Simulate for 60 trials

N=60;

MDP = mdp;

[MDP(1:N)] = deal(MDP);

MDP = spm_MDP_VB_X_tutorial(MDP);


for k = 1:60
    if (MDP(k).o(2,2) >= 1 && MDP(k).o(2,2) <= 5)
        j(k) = MDP(k).o(2,2);
    end
end
totalwin = 0; % Caculate total wins for the agent
largewin = 0;
smallwin = 0;
for k = 1:60
    if MDP(k).o(2,2) == 1
        totalwin = totalwin + 0;
    elseif MDP(k).o(2,2) == 2
        totalwin = totalwin + 4;
        largewin = largewin + 4;
    elseif MDP(k).o(2,2) == 3
        totalwin = totalwin - 4;
    elseif MDP(k).o(2,2) == 4
        totalwin = totalwin +2;
        smallwin = smallwin + 2;
    elseif MDP(k).o(2,2) == 5
        totalwin = totalwin -2;
    end 
end


simulationResults(loop).loop = loop;
simulationResults(loop).ob = ob;
simulationResults(loop).pb = (1-ob);
simulationResults(loop).winnings = j;
simulationResults(loop).totalwin = totalwin;
simulationResults(loop).smallwin = smallwin;
simulationResults(loop).largewin = largewin;

end 


%  and to show posterior beliefs and behavior:
spm_figure('GetWin','Figure 5'); clf 
spm_MDP_VB_Mood2arms(MDP);

% Save the variable to a MAT file
 save('simulationResultsTAB.mat', 'simulationResults');
