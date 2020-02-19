load letras.dat
[K,N]=size(letras);
map = [ 1 1 1;0 0 0];
for i=1:K
    x=reshape(letras(i,:),20,20)';
    image((x>0)+1)
    colormap(map)
    %pause
end
P = letras';
W = P*inv(P'*P)*P';
for i=1:K
    Yant = P(:,i);
    Ynew = sign(W*Yant);
    cont =1;
    while norm(Yant-Ynew)>0
        cont = cont+1;
        Yant = Ynew;
        Ynew = sign(W*Yant);
        
    end
    x=reshape(Ynew,20,20)';
    image((x>0)+1)
    colormap(map)
    disp(sprintf('Passos %d',cont))
    %pause
end

r=0.4;
Pr = P;
for i=1:K
    for j=1:N
        m = rand(1,1);
        if m<r
           Pr(j,i)= -Pr(j,i);
        end
    end
end

for i=1:K
    x=reshape(Pr(:,i),20,20)';
    image((x>0)+1)
    colormap(map)
    %pause
end

disp('Sincrono')
for i=1:K
    Yant = Pr(:,i);
    Ynew = sign(W*Yant);
    cont =1;    
    while norm(Yant-Ynew)>0
        cont = cont+1;
        Yant = Ynew;
        Ynew = sign(W*Yant);
    end
    x=reshape(Ynew,20,20)';
    image((x>0)+1)
    colormap(map)
    disp(sprintf('Passos %d',cont))
    pause
end

disp('Assincrono')
for i=1:K
    Yant = Pr(:,i);
    pos = ceil(rand(1,1)*N);
    Ynew = Yant;
    Yaux = sign(W*Yant);
    Ynew(pos)=Yaux(pos);
    cont =1;
    
    while norm(Yant-Yaux)>0
        cont = cont+1;
        pos = ceil(rand(1,1)*N);
        Yant = Ynew;
        Yaux = sign(W*Yant);
        Ynew(pos)=Yaux(pos);
    end
    x=reshape(Ynew,20,20)';
    image((x>0)+1)
    colormap(map)
    disp(sprintf('Passos %d',cont))
    pause
end
