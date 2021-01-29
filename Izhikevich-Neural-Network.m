Ne = 800;
Ni = 200;
re = rand(Ne,1);
ri = rand(Ni,1);
a = [0.02*ones(Ne,1); 0.02+0.08*ri];
b = [0.2*ones(Ne,1); 0.25-0.05*ri];
c = [-65+15*re.^2; -65*ones(Ni,1)];
d = [8-6*re.^2; 2*ones(Ni,1)];
S = [0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];

v = -65*ones(Ne+Ni,1);
u = b.*v;
firings = [];
for t=1:500
    I = [5*randn(Ne,1);2*randn(Ni,1)];
    fired = find(v >= 30); % 当前时刻发放脉冲的神经元index
    firings = [firings; t+0*fired,fired]; %#ok<*AGROW> % 拼接旧firing矩阵和新发放矩阵拼接起来，firings第一列是时间，1...1000,第二列是当前时刻发放神经元的index
    v(fired) = c(fired);
    u(fired) = u(fired) + d(fired);
    I = I + sum(S(:,fired),2); % 找出当前发放的所有神经元，如（1000，13），sum(S(:,fired),2)就是将1000个神经元与其相连的权重（有兴奋性，有抑制性）求和
    v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);
    v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);
    u = u + a.*(b.*v - u);
end
plot(firings(:,1),firings(:,2),'.');