%% Experiment 1 - Simulating Loss of Optimism Bias

% An Active Inference Model of the Optimism Bias 
% Elizabeth Fisher, Christopher Whyte, Jakob Hohwy
% 4/4/2024

% This code simulates the loss of the optimism bias in agents during
% development by exposing the agent to high and low arousal and positive and
% negative valence events

clear 
close all 
rng('shuffle')

% Initialize variables
variables = zeros(1, 4); % used to generate random states that the agent will be exposed to agent


for loop = 1:200
%% Level 1 Valence Arousal Model 
clear MDP_1      % Every loop is a different agent, clear the previous agent data at the start of each loop
clear MDP_optimisim
clear MDP_OB

% Prior beliefs about initial states
%--------------------------------------------------------------------------
D{1} = [1 1]';  % valence (positive, negative) 
D{2} = [1 1]';  % arousal (high, low)

T = 1; 

 
% Likelihood
%--------------------------------------------------------------------------

% Outcome
       
% A{1} Valence 
% [positive  valence, negative valence]
% high arousal
A{1}(:,:,1) = [1  0; % positive valence
               0  1]; % negative valence
% low arousal
A{1}(:,:,2) = [1  0; % positive valence
               0  1]; % negative valence
% A{2} Arousal
% [positive  valence, negative valence]
% high arousal
A{2}(:,:,1) = [1  1; % high arousal
               0  0]; % low arousal
% low arousal
A{2}(:,:,2) = [0  0; % high arousal 
               1  1]; % low arousal

% -------------------
% Control states

B{1} = eye(2);
B{2} = eye(2);

% Set up the model

mdp_1.A = A; 
mdp_1.B = B;
mdp_1.D = D;                                                                                                                                                
mdp_1.T = T;

MDP_1 = spm_MDP_check(mdp_1);

N = 52; % number of trials, 52 weeks, one year

PVHA = [1;1]; % positive valence, high arousal
PVLA = [1;2]; % positive  valence, low arousal 
NVHA = [2;1]; %  negative valence, high arosal 
NVLA = [2;2]; %  negative valence, low arousal 

% Initialize variables, this asigns random arousal and valence states for
% each agent
variables = zeros(1, 4);

% Generate random non-negative integers
for i = 1:3
    variables(i) = randi([0, 52 - sum(variables)]);
end

% The fourth variable is calculated to ensure the sum is exactly 52
variables(4) = 52 - sum(variables);

variables = variables(randperm(length(variables)));

% Assign values for each arousal valence state
poshigh = variables(1);
poslow = variables(2);
neghigh = variables(3);
neglow = variables(4);

% Set up the states for the agent
MDP_optimisim(1:N) = deal(MDP_1);
obagent = repelem([{PVHA}, {PVLA}, {NVHA}, {NVLA}], [poshigh poslow neghigh neglow]);
obagent = obagent(randperm(length(obagent))); % Shuffle the order
for l = 1:length(MDP_optimisim)
      MDP_optimisim(l).s = obagent{l};
end

MDP_OB = spm_MDP_VB_X(MDP_optimisim); % Run the model for Level 1 

clear mdp_1 A B D C T % Clear all for next agent


%% Level 2 Optimism Bias Model

 T = 52; % Time steps, each T is repsenative of a week 


% Prior beliefs about initial states
%--------------------------------------------------------------------------
 D_2{1} = [0.8 0.2]';   % Optimism [optimisim, pessimism]
 d_2{1} = [0.5, 0.5]';  % Seperate gen process from model so the agent can learn their optimisim

% Likelihood
%--------------------------------------------------------------------------
               
 %A_2{1} Valence 
 %       [optimisim, pessimism]
 A_2{1}(:,:) = [0.8  0.2; % positive valence
                0.2  0.8]; % negative valence

% A_2{2} Arousal 
% [optimisim, pessimism]

A_2{2}(:,:) = [0.5   0.5; % high arousal 
               0.5   0.5]; % low arousal 

% Seperate gen process from gen model so the agent can learn the liklihood
% mapping

a_2{1}(:,:) =  A_2{1}*1000; % agent does not learn a_2{1}

      
a_2{2}(:,:) = [0.5   0.5; % high arousal 
               0.5   0.5]; % low arousal 

% Transition probabilities: B
%--------------------------------------------------------------------------

% The agent can be a in a optimistic or pessimistic state
B_2{1} = [1 0
          0 1];

% MDP Structure
%--------------------------------------------------------------------------
   
mdp.MDP  = MDP_OB;
mdp.link = [1 0 
            0 1]; % identifies lower level state factors (rows) with higher  
                  % level observation modalities (columns). The outcomes
                  % in level are linked for each T time step. 
                

mdp.T = T;                      % number of time points
mdp.A = A_2;                    % likelihood mapping for generative process
mdp.B = B_2;                    % transition probabilities
mdp.D = D_2;                    % priors over initial states for generative process
mdp.d = d_2; 
mdp.a = a_2;

mdp = spm_MDP_check(mdp);
MDP = spm_MDP_VB_X(mdp);

%% Simulate all conditions

% Here we specify the number of trials N and use a deal function (which copies 
% the input to N outputs)

N = 5;
MDP(1:N) = deal(MDP);
MDP= spm_MDP_VB_X(MDP);

% Store model results in loop
simulationResults(loop).loop = loop;
simulationResults(loop).neghigh = variables(3);
simulationResults(loop).neglow = variables(4);
simulationResults(loop).poshigh = variables(1);
simulationResults(loop).poslow = variables(2);
simulationResults(loop).arousal = (MDP(5).a{1,2});
simulationResults(loop).optimisim =(MDP(5).d{1,1});

end 

% Save the results to a file
for loop = 1:200
simulationResults(loop).ob = normalize((simulationResults(loop).optimisim),"norm",1);  % Optimism measure
simulationResults(loop).arousalnorm = normalize((simulationResults(loop).arousal),  "norm",1); % A matrix arousal measure
end
save('simulation_results.mat', 'simulationResults');




