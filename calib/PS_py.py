# To get PROG_ID, look for your registry key "Computer\HKEY_CLASSES_ROOT\.psobj"
import time

import win32com.client

# specify your plant simulation model file path and PROG_ID for your plant simulation version. You can find PROG_ID in your registry key "Computer\HKEY_CLASSES_ROOT\.psobj"
MODEL_PATH = r"C:\Users\yxu59\files\winter2026\park\simulation\DBCsystem_v3.spp"
PROG_ID = "Tecnomatix.PlantSimulation.RemoteControl.24.4"
plant = win32com.client.DispatchEx(PROG_ID)
plant.SetVisible(False)
plant.LoadModel(MODEL_PATH)
plant.ResetSimulation(".Models.Field.EventController")

params = {
    ".Models.Field.R": 77,
    ".Models.Field.Q": 113,
    ".Models.Field.W": 27,
    ".Models.Field.CustomerLbd": 24 * 60,
    ".Models.Field.M1": 2,
    ".Models.Field.M2": 2,
}

for path, value in params.items():
    plant.SetValue(path, value)

plant.StartSimulation(".Models.Field", True)


while plant.IsSimulationRunning():
    time.sleep(1)
    result = plant.GetValue(".Models.Field.NetRevenue")
    current_time = plant.GetValue(".Models.Field.EventController.SimTime")
    
    print(current_time, result)

result = plant.GetValue(".Models.Field.NetRevenue")
plant.Quit()
print(result)
