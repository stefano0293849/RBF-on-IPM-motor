%Autore: Pota Stefano, Dipartimento di Ingegneria Elettronica, Università Degli Studi di Udine, 116983, pota.stefano@spes.uniud.it
%Data: 06/06/19
%Testato con versione di Matlab 2018.a , potrebbe dare qualche errore se usato con versioni precedenti


clear;
close all;
clc;
addpath('.\FunzioneEsternaPerOutputColorato\');                            %Aggiunta funzione esterna per avere un colore diverso nell'output della console

%****************
%Parameters
%****************
In=12;                                                                     %Valore corrente nominale                               
deltaG=0.25*In;                                                            %Distanza funzioni gaussiane, quello ottimo secondo il paper è 0.25 di In

cprintf('_Keywords', '\nCreating matrix of gaussian function..\n');

%Crea coordinate matrice gaussiana per l'hidden layer
x = -In:deltaG:In;
y = x;
[xx,yy] = meshgrid(x,y);
X(:,1)= reshape(xx,[],1);  
X(:,2)= reshape(yy,[],1);

K=size(X,1);                                                               %Numero funzioni gaussiane usate
 
B=(sqrt(K)/(2*sqrt(2)*2*In))*ones(K,1);                                    %Valore fisso definito dal paper

W=10*rand(K,2);                                                            %Valori iniziali scelti casualmente dei pesi della rete olayer
Mu=0.01;                                                                   %Costante per noi
Rs=3.4;                                                                    %Resistenza statore nota e costante
pp=2;                                                                      %Numero poli
Wme=100/pp;                                                                %Velocità elettrica
Lmg=0.221613;                                                              %Flusso magnetico
Ntraining=10;                                                              %Numero di training points

cprintf('_Keywords', '\nCreating training dataset..\n');

%****************
%Import data set for training
%****************

fid = fopen('dataIn_Rett.txt');                                            %Apertura file contenente i dati per il training e caricamente nel workspace
k=1;
tline = fgetl(fid);
while ischar(tline)
    t = extractBetween(tline,':',',');
    [a1,a2,a3,a4,a5]=deal(t{:});
    i(k,1)=str2double(a1);
    i(k,2)=str2double(a2);
    u(k,1)=str2double(a3);
    u(k,2)=str2double(a4);
    wme(k,1)=str2double(a5);
    k=k+1;
    tline = fgetl(fid);    
end
fclose(fid);

M=size(i,1);                                                               %Numero di punti a regime usati per il riferimento di corrente

%****************
% Training
%****************
    
rbf=RBFn(X,B);                                                             %Costruttore per i due layer
l=olayer(Rs,Wme,W,Mu,M,Lmg); 
figure;


cprintf('_Keywords', '\nStart training the neural network..\n');
clk = tic;                                                                 %Misuratore di performance
perc=0;                                                                    %Variabile per indicare la percentuale di learning della rete

%Collezione dell'errore per update dei pesi
for ii=1:Ntraining
    for jj=1:M
        am(jj,:)=rbf.compute(i(jj,:));                                     %Calcola l'uscita dell'hidden layer e metti i risultati nel layer di uscita
        [em(jj,:),~]=l.compute(am(jj,:),i(jj,:),u(jj,:));                  %Salve l'errore uscente dal layer di uscita, nel vettore em, contenente tutti gli errori
    end    
    
%Update Neural Network
    l=l.updatew(em,am);
        
    if ( mod(ii,Ntraining/100)==0)                                         %Update percentuale e plot grafico errore em in uscita
        plot(em);
        perc=perc+10;
        fprintf (' %0.2f%% \r', perc);
        pause(0.3);
    end
    
end    
title('Em');

%****************
%Plot dq estimated flux curves
%****************

cprintf('_Keywords', '\nPlot the estimated lambda d and q ..\n');
clear i;
clear am;
x = -12:1:4; %id domain                                                    Dominio rettangolare come definito sul file di simulazione Plecs
y = -12:1:12; %iq domain
[xx,yy] = meshgrid(x,y);
i(:,1)=reshape(xx,[],1);
i(:,2)=reshape(yy,[],1);

for jj=1:size(xx,1)*size(xx,2)                                             %Dopo la fase di traing, ricalcola valore in uscita per i flussi e plotta i grafici
    am(jj,:)=rbf.compute(i(jj,:));
    [~,lam(jj,:)]=l.compute(am(jj,:),i(jj,:),u(jj,:));
end
    
lamd=reshape(lam(:,1),[size(xx,1),size(yy,2)]);
lamq=reshape(lam(:,2),[size(xx,1),size(yy,2)]);


figure;
surf(xx,yy,lamd);                                                          %reshape e plot lamd d
xlabel('Id');
ylabel('Iq');
zlabel('$\hat{\Lambda}$d','Interpreter','latex');
title('Stima flusso asse d');
colorbar;
shading faceted;
    
figure;
surf(xx,yy,lamq);                                                          %reshape e plot lamd q
xlabel('Id');
ylabel('Iq');
zlabel('$\hat{\Lambda}$q','Interpreter','latex');
title('Stima flusso asse q');
colorbar;
shading faceted;



%****************
%Plot real d and q flux
%****************

cprintf('_Keywords', '\nPlot the ''Real'' lambda d and q ..\n');

%'Real' flux map
load('cross_saturation_D-12Q_1_12Q_-12D_1_4D.mat');
load('cross_saturation_Q-12Q_1_12Q_-12D_1_4D.mat');
i_d = cross_saturation_D(1, 2:end);
i_q = cross_saturation_Q(2:end, 1);
flux_map_D = cross_saturation_D(2:end, 2:end);
flux_map_Q = cross_saturation_Q(2:end, 2:end);
figure;
surf(i_d, i_q', flux_map_D)
xlabel('Id');
ylabel('Iq');
zlabel('${\Lambda}$q','Interpreter','latex');
title('Flusso reale asse d');
colorbar;

figure;
surf(i_d, i_q', flux_map_Q)
xlabel('Id');
ylabel('Iq');
zlabel('${\Lambda}$q','Interpreter','latex');
title('Flusso reale asse q');
colorbar;


%****************
%Plot the error
%****************

cprintf('_Keywords', '\nPlot the error of lambda d and q ..\n');

figure;
surf(i_d, i_q', ((lamd-flux_map_D)./max(flux_map_D(:))).*100)
xlabel('Id');
ylabel('Iq');
zlabel('$\epsilon_{\lambda}$d\%','Interpreter','latex','Fontsize', 18);
title('Errore asse d');

figure;
surf(i_d, i_q', ((lamq-flux_map_Q)./max(flux_map_Q(:))).*100)
xlabel('Id');
ylabel('Iq');
zlabel('$\epsilon_{\lambda}$q\%','Interpreter','latex','Fontsize', 18);
title('Errore asse q');

%****************
%Plot the slices
%****************

%D axis

cprintf('_Keywords', '\nPlot the slices of the estimated and real d and q flux ..\n');

figure;
subplot1 = subplot(2,1,1);
plot(i_d,lamd(25,:),i_d , flux_map_D(25,:) ,'LineWidth',0.5 );
ylabel({'$\hat{\Lambda}$d , ${\Lambda}$d'}, 'interpreter', 'latex');
xlabel('Id');
legend('Estimate Lambda D', 'Lambda D','Location','northwest')
title('@Iq=12A');
grid on;

subplot2 = subplot(2,1,2);
plot(i_d,lamd(13,:),i_d , flux_map_D(13,:) ,'LineWidth',0.5 );
ylabel({'$\hat{\Lambda}$d , ${\Lambda}$d'}, 'interpreter', 'latex');
xlabel('Id');
legend('Estimate Lambda D', 'Lambda D','Location','northwest')
title('@Iq=0A');
grid on;

%Q axis

figure;
subplot1 = subplot(2,1,1);
plot(i_q',lamq(:,1),i_q' , flux_map_Q(:,1) ,'LineWidth',0.5 );
ylabel({'$\hat{\Lambda}$q , ${\Lambda}$q'}, 'interpreter', 'latex');
xlabel('Iq');
legend('Estimate Lambda Q', 'Lambda Q','Location','northwest')
title('@Id=-12A');
grid on;

subplot2 = subplot(2,1,2);
plot(i_q',lamq(:,13),i_q' , flux_map_Q(:,13) ,'LineWidth',0.5 );
ylabel({'$\hat{\Lambda}$q , ${\Lambda}$q'}, 'interpreter', 'latex');
xlabel('Iq');
legend('Estimate Lambda Q', 'Lambda Q','Location','northwest')
title('@Id=0A');
grid on;


cprintf('Comments', '\n\nSimulation was finished correctly!.\n\n');
fprintf ('(t = %0.2fs).\n', toc (clk));
