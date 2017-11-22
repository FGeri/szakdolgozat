GAMMA = 0.99;
TEMPERATURE = 3;
EPISODES = 300;
LR = 0.01;
BATCH_SIZE = 32;
[x,t] = simplefit_dataset;
model = fitnet(32,'traingdm');
batch_states = [datasample(0:8,32,'Replace',true);datasample(0:8,32,'Replace',true);datasample(0:8,32,'Replace',true)];
batch_rewards = rand(3,32);

model = configure(model,batch_states,batch_rewards);
model = train(model,batch_states,batch_rewards);
log = [];
for i=1:EPISODES
%     Getting states
   batch_states = [datasample(0:8,32,'Replace',true);datasample(0:8,32,'Replace',true);datasample(0:8,32,'Replace',true)];
%     Getting actions
   y = model(batch_states);
   maxQ = max(y);
   [a,~] = find(y==maxQ);
   a_opp = (mod(floor(batch_states(1,:)/3)+1,3))';
   batch_rewards = zeros(32,1);
   batch_rewards((a-1)== mod((a_opp-1)+1,3)) = 1;
   batch_rewards(batch_rewards==0)=-1;
   batch_next_states=[((a-1)*3+a_opp-1)';batch_states(1:2,:)];
   y_=model(batch_next_states);
   next_maxQ = max(y);
   a_indeces=sub2ind(size(y),a',1:32);
   y(a_indeces)=batch_rewards'+GAMMA*next_maxQ;
   log=[log,(sum(batch_rewards)+32)/2/32*100];
   model = train(model,batch_states,y);
end
plot(1:length(log),log)