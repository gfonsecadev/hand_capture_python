from itertools import cycle
import cv2
import mediapipe as mp

#abre a camera
video = cv2.VideoCapture(0)

#mascara que criará os desenhos das conexões
handDraw = mp.solutions.drawing_utils

#configuração do modelo de reconhecimento de mãos.
handsSolutions = mp.solutions.hands
detectorHands = handsSolutions.Hands(max_num_hands=1)



#se a camera estiver aberta entraremos no loop para gerar os frames em sequencia
while(video.isOpened):
    check,frame=video.read()
    #metodo flip para inverter o frame pra apresentar de uma forma que fique na mesma posição do usuário
    #Isso é preciso pois o padrão é a imagem invertida
    frame=cv2.flip(frame,1)
    #logica para variavel check que recebe um boleano se o frame foi retornado.
    if(check):
        
        #conversão do padrão BGR do opencv para RGB aceito pelo mediapipe.
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #atribuimos ao nosso modelo o frame preparado.
        results=detectorHands.process(frameRGB)
        #atribuimos a variavel abaixo os resultado capturados pelo modelo.
        handPoints = results.multi_hand_landmarks
        typeHand=results.multi_handedness
        handSide=[]

        #capturamos a altura e a largura do frame
        height, width, _ =frame.shape
        
        coordinates = []
        if handPoints:
            #percorremos os resultados do modelo que retorna
            #fazemos um loop no resultado para capturar o objeto landmark que possui as  coordenadas da mão encontrada
            for points,hand in zip(handPoints,typeHand):
                handSide.append(hand.classification[0].label)
                #usamos nossa mascara para desenhar as conexões do objeto mão achado.
                #recebe como parametros o frame, as coordenadas do objeto achado e o driver de desenho
                handDraw.draw_landmarks(frame,points,handsSolutions.HAND_CONNECTIONS)
                #print(points)
                
                for key, allCoordinates in enumerate(points.landmark):
                    #capturamos as coordenadas x e y e multiplicamos pelo devido comprimento do eixo para obtermos a localização dos pixels
                    #coordenada x * largura e coordenada y * altura é a formula para transformar em posições de pixels
                    coordinateX, coordinateY = int(allCoordinates.x * width), int(allCoordinates.y * height)
                    #armazenamos no array como tuplas
                    coordinates.append((coordinateX,coordinateY))

                #esses são os pontos máximos das extremidades do objeto mão de acordo com a documentação
                #4= polegar, 8= indicador, 12= meio, 16= anelar, 20= mindinho 
                maxPointFingers = [8, 12, 16, 20]
                count=[0]
                print(coordinates)
                if coordinates:
                  
                  #logica para o polegar se o ponto maximo em x =4 que é a extremidade de um polegar estiver em 4 significa que ele está extendido.
                  if coordinates[4][0] < coordinates[2][0]:
                      
                      count[0] += 1
                  
                  for  side, finger in zip(cycle(handSide),maxPointFingers):
                      
                      #fazemos uma condicional que se o ponto maximo dos dedos percorridos estiverem abaixo dos pontos que se iniciam os dedos então a mão ou algum dedo está fechado.
                      #exemplo: o indicador tem o ponto máximo de 8 no eixo y, se a ponta estiver a 8-2 quer dizer que ele está fechado.
                      if (side=='Right') and  coordinates[finger][1] < coordinates[finger-2][1]:
                          count[0] += 1
                      elif(side=='Left'):
                          cv2.putText(frame,"Please use your right hand!",(15,65), cv2.FONT_HERSHEY_SIMPLEX,1 ,(25,25,255) , 2 ,cv2.LINE_AA)                      
                if count[0]:
                 cv2.putText(frame,str(count),(15,65), cv2.FONT_HERSHEY_SIMPLEX,2 ,(25,25,255) , 3)

                #print(count)
    cv2.imshow("video",frame)
    cv2.waitKey(1)