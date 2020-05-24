import numpy as np
import xlwings as xw
import pandas as pd
class design_env:
  def __init__(self, ruta):
    self.ruta = ruta    
    self.state_df = xw.Book(self.ruta).sheets['State'].range('C2:C46').value#.options(ndim=2)
    self.verif_df = xw.Book(self.ruta).sheets['Verifications'].range('B2:B21').value
    self.reward = [] #list of lists



#Setting a value on a df range
  def apply_action(self, action):#recibe un vector "action" one-hot encoder
    wb = xw.Book(self.ruta)
    hoja_aux1 = wb.sheets['aux1']
    index = np.argmax(action)#obtiene posicion de la accion
    
    if index == 0:
        hoja_aux1.range('B1').value = hoja_aux1.range('B1').value + 0.05

    if index == 1:
      if self.state_df[12] >= 0.3:
        hoja_aux1.range('B1').value = hoja_aux1.range('B1').value - 0.05  

    if index == 2:
      hoja_aux1.range('B2').value = hoja_aux1.range('B2').value + 0.1
    if index == 3:
      hoja_aux1.range('B2').value = hoja_aux1.range('B2').value - 0.1  

    if index == 4:
      hoja_aux1.range('B3').value = hoja_aux1.range('B3').value + 0.1
    if index == 5:
        if hoja_aux1.range('B3').value >= hoja_aux1.range('B5').value + 0.1:
            hoja_aux1.range('B3').value = hoja_aux1.range('B3').value - 0.1
        
    if index == 6:
      hoja_aux1.range('B4').value = hoja_aux1.range('B4').value + 0.1
    if index == 7:
        if hoja_aux1.range('B4').value >=hoja_aux1.range('B6').value + 0.1:
            hoja_aux1.range('B4').value = hoja_aux1.range('B4').value - 0.1  

    if index == 8:
      hoja_aux1.range('B5').value = hoja_aux1.range('B5').value + 0.05
    if index == 9:
      if hoja_aux1.range('B5').value >= 0.45:
        hoja_aux1.range('B5').value = hoja_aux1.range('B5').value - 0.05

    if index == 10:
      hoja_aux1.range('B6').value = hoja_aux1.range('B6').value + 0.05
    if index == 11:
      if hoja_aux1.range('B6').value >= 0.45:
       hoja_aux1.range('B6').value = hoja_aux1.range('B6').value - 0.05    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # if index == 0:
    #   self.state_df[12] = self.state_df[12] + 0.05
    # if index == 1:
    #   if self.state_df[12] >= 0.3:
    #     self.state_df[12] =  self.state_df[12] - 0.05  

    # if index == 2:
    #   self.state_df[13] = self.state_df[13] + 0.1
    # if index == 3:
    #   self.state_df[13] = self.state_df[13] - 0.1  

    # if index == 4:

    #   self.state_df[14] = self.state_df[14] + 0.1
    # if index == 5:
    #     if self.state_df[14] >=self.state_df[16] + 0.1:
    #         self.state_df[14] = self.state_df[14] - 0.1  

    # if index == 6:
    #   self.state_df[15] = self.state_df[15] + 0.1
    # if index == 7:
    #     if self.state_df[15] >=self.state_df[17] + 0.1:
    #         self.state_df[15] = self.state_df[15] - 0.1  

    # if index == 8:
    #   self.state_df[16] = self.state_df[16] + 0.05
    # if index == 9:
    #   if self.state_df[16] >= 0.45:
    #     self.state_df[16] = self.state_df[16] - 0.05

    # if index == 10:
    #   self.state_df[17] = self.state_df[17] + 0.05
    # if index == 11:
    #   if self.state_df[17] >= 0.45:
    #     self.state_df[17] = self.state_df[17] - 0.05
 
    #ahora debemos hacer los cambios en la planilla, luego recoger la info
    #de reward y new state dados estos cambios
    # rewriting the new df on excel

    # wb = xw.Book(self.ruta)
    # hoja_aux1 = wb.sheets['aux1']
    # hoja_aux1.range('B1:B6').options(index=False, header=False).value = pd.DataFrame(self.state_df[12:18]).values#.options(ndim=2)
   

  def get_state(self):
    wb = xw.Book(self.ruta)
    new_state = wb.sheets['State'].range('C2:C46').value   
    return new_state

  def get_reward(self):
    wb = xw.Book(self.ruta)
    new_reward = wb.sheets['Verifications'].range('B2:B21').value
    weights = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.24]
    weighted_avg = np.average(new_reward, weights=weights)
    self.reward.append(new_reward)
    #sum_reward = sum(new_reward) # simplementela suma de las verificaciones...
                                 #podriamos darle mas peso al peso.fund por ejemplo
    return weighted_avg#sum_reward
  
  def game_over(self):
      if -1 in self.reward[-1][0:20]: # any -1 in las reward append, los busca sin considerar el peso.
          return True
      else:
          return False
  def close(self):
      wb = xw.Book(self.ruta)
      wb.app.quit()
  
  def set_start(self):
      wb = xw.Book(self.ruta)
      wb.sheets['aux1'].range('B1:B6').options(ndim=2).value = wb.sheets['aux1'].range('H1:H6').options(ndim=2).value
      return 
          





