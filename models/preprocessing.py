import pandas as pd

def preprocess_data(df):


    mapping_admission_type_id ={
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped"
    }


    mapping_discharge_disposition_id = {
        1: "Discharged to home",
        2: "Discharged/transferred to another short term hospital",
        3: "Discharged/transferred to SNF",
        4: "Discharged/transferred to ICF",
        5: "Discharged/transferred to another type of inpatient care institution",
        6: "Discharged/transferred to home with home health service",
        7: "Left AMA",
        8: "Discharged/transferred to home under care of Home IV provider",
        9: "Admitted as an inpatient to this hospital",
        10: "Neonate discharged to another hospital for neonatal aftercare",
        11: "Expired",
        12: "Still patient or expected to return for outpatient services",
        13: "Hospice / home",
        14: "Hospice / medical facility",
        15: "Discharged/transferred within this institution to Medicare approved swing bed",
        16: "Discharged/transferred/referred another institution for outpatient services",
        17: "Discharged/transferred/referred to this institution for outpatient services",
        18: "NULL",
        19: "Expired at home. Medicaid only, hospice.",
        20: "Expired in a medical facility. Medicaid only, hospice.",
        21: "Expired, place unknown. Medicaid only, hospice.",
        22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
        23: "Discharged/transferred to a long term care hospital",
        24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
        25: "Not Mapped",
        26: "Unknown/Invalid",
        27: "Discharged/transferred to a federal health care facility",
        28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
        29: "Discharged/transferred to a Critical Access Hospital (CAH)",
        30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    }

    mapping_admission_source_id = {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from a hospital",
        5: "Transfer from a Skilled Nursing Facility (SNF)",
        6: "Transfer from another health care facility",
        7: "Emergency Room",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from critical access hospital",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Extramural Birth",
        15: "Not Available",
        17: "NULL",
        18: "Transfer From Another Home Health Agency",
        19: "Readmission to Same Home Health Agency",
        20: "Not Mapped",
        21: "Unknown/Invalid",
        22: "Transfer from hospital inpt/same fac reslt in a sep claim",
        23: "Born inside this hospital",
        24: "Born outside this hospital",
        25: "Transfer from Ambulatory Surgery Center",
        26: "Transfer from Hospice",
    }
    # Drop encounter_id
    df = df.drop(['encounter_id'], axis=1)
    df = df.replace(['?','NULL'], pd.NA)

    # Lowercase all Column Names
    df.columns = df.columns.str.lower() 

    # Edit the readmitted column to be a binary target. We are focused on patients readmitted within 30 days.
    df["readmitted"] = df["readmitted"].replace({">30": 0, "<30": 1, "NO": 0})

    df.rename(columns={"readmitted":"target"},inplace=True)
    df["target"].astype(int)
    df["admission_type_id"] = df["admission_type_id"].map(mapping_admission_type_id)
    df["discharge_disposition_id"] = df["discharge_disposition_id"].map(mapping_discharge_disposition_id)
    df["admission_source_id"] =df["admission_source_id"].map(mapping_admission_source_id)
    return df


def import_data():
    print("Importing train/validation data...")
    Xt = pd.read_csv("modeling_data/train.csv")
    yt = Xt["target"]
    Xt.drop(columns=["target"],inplace=True)
    Xv = pd.read_csv("modeling_data/validation.csv")
    yv = Xv["target"]
    Xv.drop(columns=["target"],inplace=True)

    Xt.drop(columns=["patient_nbr"],inplace=True)
    Xv.drop(columns=["patient_nbr"],inplace=True)
    
    print("Finished the import.")
    return Xt,yt,Xv,yv