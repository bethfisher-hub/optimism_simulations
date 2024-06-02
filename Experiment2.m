%% Experiment 2 - Optimism Bias Belief Updating Task

% An Active Inference Model of the Optimism Bias 
% Elizabeth Fisher, Christopher Whyte, Jakob Hohwy
% 4/4/2024

% Belief Updating Task simulation to measure the optimism bias. Task from
% Sharot 2011


clear 
close all 

for group = 1:9
% Generate 70 random numbers for the beliefs about good and bad outcomes,
% set beliefs at this range as most similar to the list from Sharot et
% al.,2011
go = 0.2 + (0.7 - 0.2) * rand(1, 70);
outcomes = randi([1, 2], 1, 70);
% After a parameter sweep the prec value that best matches the literature 

%% Level 1 beliefs about concerns 
% Loop through levels of optimism  
for j = 1:10
ob = j/10;
% Priors over the initial states
D{2} = [ob (1-ob)]'; % Optimism, pessimism
d{2} = D{2}*100;  % Agent does not learn it's optimism

% Optimistic belief updating, generative process
                %good outcome, bad outcome  
A{1}(:,:,1) =   [1 0    % good outcome
                 0 1];   % bad outcome 
% Pesmesitc belief 
A{1}(:,:,2) = [1 0      % good outcome 
               0 1];    % bad outcome 
% Generative model

%Optimisitc belief updating
a{1}(:,:,1) =    [0.9   0.1    % good outcome
                  0.4   0.6];  % bad outcome 
% Pesmesitc belief updating
a{1}(:,:,2) = [0.6  0.4  % good outcome
               0.1  0.9];  % bad outcome 
% No state changes 
B{1}  = [1 0
         0 1];

B{2}  = [1 0
         0 1];

% Trial is one update
T = 1;

for i = 1:70
% Loop through the 70 beliefs about good v bad outcomes
D{1} = [go(i) (1-go(i))]'; % belief about good and bad outcome 
d{1} = [go(i) (1-go(i))]';


mdp.T = T;                    % Number of time steps
mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.D = D;                    % priors over initial states
mdp.o = outcomes(i);          % set outcomes 
mdp.d = d;
mdp.a = a;
         
mdp = spm_MDP_check(mdp);
MDP = spm_MDP_VB_X(mdp);

% Save updated belief 
posterior{i} = MDP.X{1,1};
goodupdatepos(i) = [posterior{i}(1,1)]; 
badupdatepos(i) = [posterior{i}(2,1)]; 
updatedbelief{i} = normalize((MDP.d{1,1}),"norm",1);%normalise this belief 
goodupdate(i) = [updatedbelief{i}(1,1)]; 
badupdate(i) = [updatedbelief{i}(2,1)]; 


end
% Caluculate difference in beliefs for good and bad news
for k = 1:70
if outcomes(k) == 1
    beliefdifgoodpos(k) =  goodupdatepos(k) - go(k);
    beliefdifgood(k) =  goodupdate(k) - go(k);
    beliefdifbadpos(k) = "NA";
    beliefdifbad(k) = "NA";
elseif outcomes(k) == 2
    beliefdifbadpos(k) = badupdatepos(k) - (1-go(k));
    beliefdifbad(k) = badupdate(k) - (1-go(k));
    beliefdifgoodpos(k) = "NA";
    beliefdifgood(k) = "NA";

end 
end  
% This is needed for saving the first agent
if ob == 0.1 
    if isstring(beliefdifgood)
    beliefdifgood = str2double(beliefdifgood);
    beliefdifgoodpos = str2double(beliefdifgoodpos);
    end
    if isstring(beliefdifbad)
    beliefdifbad = str2double(beliefdifbad);
    beliefdifbadpos = str2double(beliefdifbadpos);
    end

end
% Caculate the average belief updating for good and bad news
    avebadupdatepos = nanmean(beliefdifbadpos)*100;
    avegoodupdatepos = nanmean(beliefdifgoodpos)*100;
    avebadupdate = nanmean(beliefdifbad)*100;
    avegoodupdate = nanmean(beliefdifgood)*100;

simulationResults_bu(j).ob = ob;
simulationResults_bu(j).go = go;
simulationResults_bu(j).goodupdatepos = goodupdatepos;
simulationResults_bu(j).badupdatepos = badupdatepos;
simulationResults_bu(j).beliefdifbad =  beliefdifbad;
simulationResults_bu(j).beliefdifgood =  beliefdifgood;
simulationResults_bu(j).beliefdifbadpos =  beliefdifbadpos;
simulationResults_bu(j).beliefdifgoodpos =  beliefdifgoodpos;
simulationResults_bu(j).avebadupdate =  avebadupdate;
simulationResults_bu(j).avegoodupdate = avegoodupdate;
simulationResults_bu(j).avebadupdatepos =  avebadupdatepos;
simulationResults_bu(j).avegoodupdatepos = avegoodupdatepos;

end


end 

% Save the variable to a MAT file
 save('simulationResults_bp.mat', 'simulationResults_bu');





