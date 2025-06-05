import numpy as np
import pandas as pd
import pickle

def convert_theia_to_myo_data(theia_table_path, output_path):
    record_time_sec = 30
    df = pd.read_csv(theia_table_path,
                            skiprows=lambda x: x in range(0, record_time_sec*1000 + 8), # skip force plate data and 10 rows of header
                            encoding_errors='ignore',
                            on_bad_lines='skip',
                            low_memory=False)
    
    df.iloc[0] = df.iloc[0].ffill()
    for col in df.columns:
        df.loc[1, col] = str(df.loc[0, col]) + " " + str(df.loc[1, col])+ " "+str(df.loc[2, col])
    df = df[1:].drop(index=2).reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df[1:].iloc[1:]
    for col in df.columns:
        if " deg" in col:
            df[col] = df[col].astype(float) * 0.0174533
        if " mm" in col:
            df[col] = df[col].astype(float) * 0.001
    df.columns = [col[11:] for col in df.columns]  

    myo_data = {'qpos': pd.DataFrame(), 'qvel': pd.DataFrame()}

    def modify(type, myo_var, theia_var, flip = 1):
        if type == 'p':
            myo_data['qpos'][myo_var] = flip * df[theia_var]
        elif type == 'v':
            myo_data['qvel'][myo_var] = flip * df[theia_var]
        else:
            print("invalid type")

    # modify('v', "hip_flexion_l", "LeftHipAngles_Theia X' deg/s")
    # modify('v',"hip_flexion_r", "RightHipAngles_Theia X' deg/s")
    modify('p',"hip_flexion_l", "LeftHipAngles_Theia X deg")
    modify('p',"hip_flexion_r", "RightHipAngles_Theia X deg")

    # modify('v', "hip_adduction_l", "LeftHipAngles_Theia Y' deg/s")
    # modify('v', "hip_adduction_r", "RightHipAngles_Theia Y' deg/s")
    modify('p', "hip_adduction_l", "LeftHipAngles_Theia Y deg")
    modify('p', "hip_adduction_r", "RightHipAngles_Theia Y deg")

    # modify('v',"hip_rotation_l", "LeftHipAngles_Theia Z' deg/s")
    # modify('v',"hip_rotation_r", "RightHipAngles_Theia Z' deg/s")
    modify('p',"hip_rotation_l", "LeftHipAngles_Theia Z deg")
    modify('p',"hip_rotation_r", "RightHipAngles_Theia Z deg")

    # modify('v',"knee_angle_l", "LeftKneeAngles_Theia X' deg/s")
    # modify('v',"knee_angle_r", "RightKneeAngles_Theia X' deg/s")
    modify('p',"knee_angle_l", "LeftKneeAngles_Theia X deg")
    modify('p',"knee_angle_r", "RightKneeAngles_Theia X deg")

    # modify('v',"ankle_angle_l", "LeftAnkleAngles_Theia X' deg/s")
    # modify('v',"ankle_angle_r", "RightAnkleAngles_Theia X' deg/s")
    modify('p',"ankle_angle_l", "LeftAnkleAngles_Theia X deg")
    modify('p',"ankle_angle_r", "RightAnkleAngles_Theia X deg")

    # modify('v',"mtp_angle_l", "l_toes_4X4 RX' deg/s", -1)
    # modify('v',"mtp_angle_r", "r_toes_4X4 RX' deg/s", -1)
    modify('p',"mtp_angle_l", "l_toes_4X4 RX deg", -1)
    modify('p',"mtp_angle_r", "r_toes_4X4 RX deg", -1)

    # modify('v',"subtalar_angle_l", "LeftAnkleAngles_Theia Y' deg/s")
    # modify('v',"subtalar_angle_r", "RightAnkleAngles_Theia Y' deg/s")
    modify('p',"subtalar_angle_l", "LeftAnkleAngles_Theia Y deg")
    modify('p',"subtalar_angle_r", "RightAnkleAngles_Theia Y deg")

    # Save the dictionary using pickle
    with open(output_path, 'wb') as file:
        pickle.dump(myo_data, file)


if __name__ == '__main__':
    convert_theia_to_myo_data('AB01_Jimin_1p0mps_1.csv', 'myo_data.pkl')