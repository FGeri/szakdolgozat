%Missing exploration strategy,decreasing the alfa and the d values,
%
alfa=0.5;
d=0;
%állapot tér

%1 Kő vs kő
%2 Kő vs papír
%3 Kő vs olló
%4 Papír vs kő
%5 Papír vs papír
%6 Papír vs olló
%7 Olló vs kő
%8 Olló vs papír 
%9 Olló vs olló
%10 nem volt még
%1: k-1. kör,k-2. kör, k-3. kör, k-4. kör, k-5. kör,
startingState= [10 10 10];
aToS=[1 2 3; 4 5 6; 7 8 9];
s=startingState;
%Közös actionspace, 1 kő,2 papír, 3 olló
a=[1 2 3];
c=zeros(10,10,10);
q=zeros(10,10,10,3);
baseIndex=10^length(s);
sIndex=sub2ind(size(c),s(1),s(2),s(3));

botPolicy=[0.60 0.20 0.20];

rewardOfPlayerOne=[0 -10 10; 10 0 -10; -10 10 0];
policy=ones(10,10,10,3)/length(a);
avpolicy=policy;
r1=0;
r2=0;
maxQ=0;
indexOfMaxQ=1;
policyChange=0;
%epsilon greedy
epsilon=0.05;
gamma=0.9;
explore = (rand<epsilon);
data=[];
moves= [];
sampleTime=100;
timer=0;
interations=18000;
wonGames=1;
drawGames=1;
lostGames=1;

figure


for i=1:interations

	epsilon=0.3/(1+i/(15));
	sIndex=sub2ind(size(c),s(1),s(2),s(3));
	alfa=1/(10+(i/10000));
	dw=1/(100+(i/100));
	dl=2*dw;

	randomDecisionOne=rand;
	randomDecisionTwo=rand;



	%Taking a step with player1

	if explore
		actionOfPlayerOne=datasample(a,1);
	else
		%Taking a step with player1
		if (randomDecisionOne<=policy(sIndex+(1-1)*baseIndex))
			actionOfPlayerOne=1;
		elseif (randomDecisionOne<=(policy(sIndex+(1-1)*baseIndex))+policy(sIndex+(2-1)*baseIndex))
			actionOfPlayerOne=2;
		else
			actionOfPlayerOne=3;
		end
	end


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%Taking a step with player2
	%Take the best step for the previous situation


	if (s(1)~=10)
		actionOfPlayerTwo=mod(floor((s(1)-1)/3)+1,3)+1;
	
	else
		actionOfPlayerTwo=datasample([1 2],1);
	end

	%Taking a step based on a possibility distribution
	%	if (randomDecisionTwo<=botPolicy(1))
	%		actionOfPlayerTwo=1;
	%	elseif (randomDecisionTwo<=botPolicy(1)+botPolicy(2))
	%		actionOfPlayerTwo=2;
	%	else
	%		actionOfPlayerTwo=3;
	%	end




	moves=[moves; actionOfPlayerOne actionOfPlayerTwo];

	%Observing the reward
	r1=rewardOfPlayerOne(actionOfPlayerOne,actionOfPlayerTwo);
	r2=-rewardOfPlayerOne(actionOfPlayerOne,actionOfPlayerTwo);
	nextState=[aToS(actionOfPlayerOne,actionOfPlayerTwo) s(1,1:end-1)];
	
	%Updating our Q values
	q(sIndex+(actionOfPlayerOne-1)*baseIndex)=q(sIndex+(actionOfPlayerOne-1)*baseIndex)+alfa*(r1+gamma*max(q(nextState(1),nextState(2),nextState(3),1:end))-q(sIndex+(actionOfPlayerOne-1)*baseIndex));
	
	%Updating our average policy
	c(sIndex)=c(sIndex)+1;
	
	%c(10,10,10)=1234;
	[maxQ,indexOfMaxQ]=max(q(s(1),s(2),s(3),1:end));


	%policy(sIndex+(j-1)*baseIndex)-avpolicy(sIndex+(j-1)*baseIndex);

	for j = a
		avpolicy(sIndex+(j-1)*baseIndex)=avpolicy(sIndex+(j-1)*baseIndex)+1/c(sIndex)*(policy(sIndex+(j-1)*baseIndex)-avpolicy(sIndex+(j-1)*baseIndex));
	end

	if sum(policy(s(1),s(2),s(3),:).*q(s(1),s(2),s(3),:)) > sum(avpolicy(s(1),s(2),s(3),:).*q(s(1),s(2),s(3),:))
		d=dw;
	else
		d=dl;
	end

	%Step our policy closer to the optimal policy
	policyChange=0;
	for j = a
		if (sum(find(indexOfMaxQ==j))==0 && policy(sIndex+(j-1)*baseIndex)>d)
			policyChange=policyChange+d;
			policy(sIndex+(j-1)*baseIndex)=policy(sIndex+(j-1)*baseIndex)-d;
		end
	end

	for j = a
		if (sum(find(indexOfMaxQ==j))~=0)
			policy(sIndex+(j-1)*baseIndex)=policy(sIndex+(j-1)*baseIndex)+policyChange/length(indexOfMaxQ);
		end
	end

	%Change states
	s=nextState;


	%Sampling
	if r1==10
			wonGames=wonGames+1;
	end
	if r1==0
			drawGames=drawGames+1;
	end
	if r1==-10
			lostGames=lostGames+1;
	end

	if (timer==0)
		
		data=[data;alfa dw wonGames/sampleTime*100 drawGames/sampleTime*100 lostGames/sampleTime*100];

		pie([wonGames+1 drawGames+1 lostGames+1]);
		labels = {'Won','Draw','Lost'};
		legend(labels,'Location','southoutside','Orientation','horizontal')
		drawnow;		
		wonGames=0;
		drawGames=0;
		lostGames=0;
	end
	timer=timer+1;
	timer=mod(timer,sampleTime);
end
t = linspace(1,length(data(:,1)),length(data(:,1)));
y1=(data(:,1))';
y2=data(:,2)';



p1 = subplot(1,1,1); % top subplot

annotation('textbox', [0 0.9 1 0.1], ...
    'String', 'WOLF-PHC', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center')


title(p1,'WOLF-PHC');
plot(p1,t,data(:,3));
xlabel(p1,'Episodes');
ylabel(p1,'Won games (%)');