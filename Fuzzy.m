% mudar para "Fuzzy3i" para processar 3 atrasos, "Fuzzy2i" para 2 atrasos
res = readfis('Fuzzy3i');
% mudar para serieX_dadosYe.txt, sendo X a série, e Y o nº de entradas
load ('popentrada.txt');
% mudar para serieX_dadosYe.txt, sendo X a série, e Y o nº de entradas
load ('popsaida.txt');

% remove as tendências (opcional)
popentrada = detrend(popentrada);
popsaida = detrend(popsaida);

EQM = 0;
EQM_ant = 0;
EQM_melhor = 0;
comb = 1;

for m=0.1:0.1:4 %centro
    for sig = 0.1:0.1:4 %desvio
        for i = 1:3 %mudar o valor, com base no numero de entradas - for i = 1:número de entradas
            x = res.input(i);
            for j = 1:3
                x.mf(j).params(1) = m;
                x.mf(j).params(2) = sig;
            end
            res.input(i) = x;
            saida = evalfis(popentrada,res);
            erro = saida - popsaida;
            EQM = erro' * erro/length(erro);
            comb = comb+1;
            if (EQM < EQM_ant)
                save('melhor','res');
                EQM_melhor = EQM;
            end
            EQM_ant = EQM;
        end        
    end
end

disp(EQM_melhor);
disp(comb);