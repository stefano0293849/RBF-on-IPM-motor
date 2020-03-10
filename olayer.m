classdef olayer
    
    properties                                                             
        Rs                                                                 %Resistenza statore
        Wme                                                                %Velocità elettrica motore
        W                                                                  %Vettore di pesi grande K x 2 ( per d e q), prima colonna di W è wd seconda è wq
        K                                                                  %Numero di pesi usati per asse d e q
        Mu                                                                 %Fattore di learning dell'algoritmo,per noi è costante
        M                                                                  %Numero di dati in ingresso per aggiornamento pesi
        Lmg                                                                %Flusso magnetico 

    end
    methods
        
      %Constructor
      function obj=olayer(rs,wme,w,mu,m,lmg) 
            obj.Rs=rs;
            obj.K=size(w,1);
            obj.Wme=wme;  
            obj.W=w;
            obj.Lmg=lmg;                                                   %non aggiorno wbd ed wbq per tener conto dell'offset,visto che lo fa da solo l'algoritmo 
            obj.Mu=mu;
            obj.M=m; 
       end
       
      %Funzione per calcolare l'errore in uscita dall'output layer
      function [e,lam] = compute(obj,a,i,u)                                %uscita errore, i(:,1) = id ,i(:,2) = iq
                   fi=(a').*(obj.W);                                       %w0=wd,w1=wq   
                   fi=sum(fi);                                             %sommato su colonne per trovare fid=f(1,1) e fiq=f(1,2)                              
                   ed=u(1,1) - ((fi(1,2).*(-obj.Wme)) + (obj.Rs*i(1,1)));  %calcolo errore dei due scalari ed,eq
                   eq=u(1,2) - ((fi(1,1).*(+obj.Wme)) + (obj.Rs*i(1,2)));  
                   e=[ed,eq];
                   lam=[fi(1,1),fi(1,2)];

       end
      
      %Funzione per aggiornare i pesi del layer di uscita dopo M valori in
      %ingresso del riferimento di corrente
      function obj = updatew(obj,em,am)                                    %em=matrice contenente gli M errori dei vari riferimenti di corrente usati,am=contiene MxK uscite dell'hidden layer,usata per Jd,Jq
          I= eye(obj.K);
          obj.W(:,1)=obj.W(:,1)-((((-obj.Wme*(am)).')*(-obj.Wme*(am))+obj.Mu*I)^-1)*((-obj.Wme*(am)).')*(em(:,2)) ; %Jd=(-obj.Wme*(am));
          obj.W(:,2)=obj.W(:,2)-((((+obj.Wme*(am)).')*(+obj.Wme*(am))+obj.Mu*I)^-1)*((+obj.Wme*(am)).')*(em(:,1)) ; %Jq=(+obj.Wme*(am));
      end  
            
    end          
end