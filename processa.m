% Ultima revisao 25/03/99
% [Ew,dEw] = processa2(X,S,A,B,C,n,m,N)
% para rede MLP com recorrencias externas de ordem generica >= 1
% rede contendo uma camada oculta
% saidas com funcoes de ativacao lineares
% n = [numero de unidades ocultas , numero de saidas]
% m: numero de entradas externas
% L: ordem do filtro que tem cada uma das saidas como entrada

function [Ew,dEw] = processa2(X,Saida,P,A,B,C,n,m,N,L)
X = [X ones(N,1)];  
S = zeros(n(1),1);  % somas para unidades ocultas
Z = zeros(n(1),1);  % saidas das unidades ocultas
Y = zeros(n(2),1);  % saidas da rede

Yold = zeros(n(2)*L,1);                    % difere de processa.m
dYdCold = zeros(n(2)*L,n(2)*(n(1)+1));     % difere de processa.m
dZdC = zeros(n(1),n(2)*(n(1)+1));
MatZ = zeros(n(2),n(2)*(n(1)+1));
dYdC = zeros(n(2),n(2)*(n(1)+1));
dJdC = zeros(1,n(2)*(n(1)+1));
dJTdC = zeros(1,n(2)*(n(1)+1));

MatYold = zeros(n(1),n(1)*n(2)*L);         % difere de processa.m
dYdAold = zeros(n(2)*L,n(1)*n(2)*L);       % difere de processa.m
dZdA = zeros(n(1),n(1)*n(2)*L);            % difere de processa.m
dYdA = zeros(n(2),n(1)*n(2)*L);            % difere de processa.m
dJdA = zeros(1,n(1)*n(2)*L);               % difere de processa.m
dJTdA = zeros(1,n(1)*n(2)*L);              % difere de processa.m

MatU = zeros(n(1),n(1)*(m+1));              
dYdBold = zeros(n(2)*L,n(1)*(m+1));        % difere de processa.m
dZdB = zeros(n(1),n(1)*(m+1));
dYdB = zeros(n(2),n(1)*(m+1));
dJdB = zeros(1,n(1)*(m+1));
dJTdB = zeros(1,n(1)*(m+1));

veterro = zeros(N,1);

h = n(1);
c = n(2);
hmaior = h+1;
for t = 1:N,
   
 % propagacao direta
   U = X(t,:)';      % entrada externa atual
   d = Saida(t,:)';      % saidas desejadas
   S = A*Yold + B*U;
   Z = tanh(S);  % entradas para a camada de saida
   Y = C*[Z;1];
   erro = Y-d;
   veterro(t) = (P(t).*erro)'*erro;
   dZdS = (1.0-Z.*Z);
   DiagdZdS = diag(dZdS);

 % calculo das derivadas de J em relacao a C(l,k)
       
   dZdC = DiagdZdS*A*dYdCold;      %calculo de dZ/dC(l,k)
   for k = 1:c,
     MatZ(k,(k-1)*hmaior+1:k*hmaior) = [Z;1]';
   end
   dYdC = C(:,1:h)*dZdC + MatZ;     %calculo de dY/dC(l,k)
   dJdC = (P(t).*erro)'*dYdC;               %claculo de dJ/dC
   dJTdC = dJTdC + dJdC;            %gradiente total
 
 % calculo das derivadas de J em relacao a A(r,k)
   
   for j = 1:h,         % comp bloco = c*L
     MatYold(j,(j-1)*c*L+1:j*c*L) = Yold';    % difere de processa.m
   end
   dZdA = DiagdZdS*(A*dYdAold + MatYold);   %calculo de dZ/dA
   dYdA = C(:,1:h)*dZdA;                   %calculo de dY/dA
   dJdA = (P(t).*erro)'*dYdA;                      %calculo de dJ/dA
   dJTdA = dJTdA + dJdA;                   %gradiente total
   
% calculo das derivadas de J em relacao a B(r,i)

   for i = 1:h,
     MatU(i,(i-1)*(m+1)+1:i*(m+1)) = U';
   end
   dZdB = DiagdZdS*(A*dYdBold + MatU);      %calculo de dZ/dB
   dYdB = C(:,1:h)*dZdB;                    %calculo de dY/dB
   dJdB = (P(t).*erro)'*dYdB;                       %calculo de dJdB
   dJTdB = dJTdB + dJdB;                    %gradiente total

% atualizacoes

   Yold(2:end) = Yold(1:end-1);
   Yold(1:L:(c-1)*L+1) = Y;
   dYdCold(2:end,:) = dYdCold(1:end-1,:);
   dYdCold(1:L:(c-1)*L+1,:) = dYdC;
   dYdAold(2:end,:) = dYdAold(1:end-1,:);
   dYdAold(1:L:(c-1)*L+1,:) = dYdA; 
   dYdBold(2:end,:) = dYdBold(1:end-1,:);
   dYdBold(1:L:(c-1)*L+1,:) = dYdB;

end

Ew = 0.5*sum(veterro);
dEw = [dJTdA  dJTdB  dJTdC]';



