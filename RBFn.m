classdef RBFn
    
    properties
        X                                                                  %Vettore contenente le coordinate delle funzioni gaussiane K x 2
        B                                                                  %Vettore contenente i bias della rbf K x 1
        K                                                                  %Numero funzioni gaussiane usate
    end
    methods
        
      %Contructor
      function obj=RBFn(x,b)                                               %costruttore, x definisce le coordinate delle funzioni gaussiane (Kx2) ,b è il vettore di bias (Kx1)
            obj.X=x;                                                      
            obj.B=b;                                                       
            %obj.B(end)=0;
            %%set dell'ultimo bias a zero, per tenere conto che a corrente
            %%nulla l'uscita del flusso d vale lambdamg,l'ho rimossa per chiarezza
            %%concettuale ma funziona in entrambi i modi,basta
            %%decommentarla
            obj.K=size(b,1);
      end  
      
      %Funzione che calcola l'uscita dell'hidden layer RadialBasisNetwork
      function a = compute(obj,i)                                          %i è il vettore di corrente in ingresso della rete         
                                                                           %Aggiungo nella prima riga delle coordinate delle funzioni gaussiane, il vettore di corrente, per come funziona la funzione dist         
          obj.X=[0 ,0; obj.X ];                                            %Per far questo inizializzo a zero la prima riga per poi modificarla successivamente con il valore di corrente in ingresso della rete
          obj.X(1,:)=i;
          dist=pdist(obj.X);                                               %Calcolo la distanza e prendo i primi K elementi che sono le K distanze riferite rispetto il vettore di corrente
          dist=(dist(1,1:obj.K))';
          n=dist.*obj.B;                                                   %Moltiplicato per il bias e per una funzione gaussiana
          a=(gaussmf(n,[sqrt(2)/2 0])).';                                  
     end
            
    end          
end
    
    
