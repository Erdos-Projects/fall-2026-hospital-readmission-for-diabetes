import pandas as pd
import numpy as np

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


def group_diagnoses(df):
    ranges = {"Circulatory": (390, 459), "Respiratory": (460, 519),
        "Digestive": (520, 579), "Genitourinary": (580, 629),
        "Musculoskeletal": (710, 739), "Neoplasms": (140, 239), "Injury": (800, 999)}
    
    symptoms = {785: "Circulatory", 786: "Respiratory", 787: "Digestive", 788: "Genitourinary"}

    def categorize(value):
        if pd.isna(value):
            return "Missing"

        code = str(value).strip().upper()
        if code in {"", "?", "NAN", "NONE", "MISSING"}:
            return "Missing"

        # Supplementary V/E codes belong to the broad Other category.
        if code.startswith(("V", "E")):
            return "Other"

        try:
            number = float(code)
            if not 1 <= number < 1000:
                return "Other"
            code = int(number)  # 250.13 -> 250, 459.9 -> 459
        except ValueError:
            return "Other"

        if code == 250:
            return "Diabetes"
        if code in symptoms:
            return symptoms[code]
        for group, (lower, upper) in ranges.items():
            if lower <= code <= upper:
                return group
        return "Other"

    result = df.copy()
    for column in ("diag_1", "diag_2", "diag_3", "diaga_3"):
        if column in result.columns:
            result[column] = result[column].map(categorize)

    return result


def group_medications(df):
    groups = {
        "biguanide": ["metformin"],
        "meglitinide": ["repaglinide", "nateglinide"],
        "sulfonylurea_gen1": ["chlorpropamide", "acetohexamide", "tolbutamide", "tolazamide"],
        "sulfonylurea_gen2": ["glimepiride", "glipizide", "glyburide"],
        "thiazolidinedione": ["pioglitazone", "rosiglitazone", "troglitazone"],
        "alpha_glucosidase_inhibitor": ["acarbose", "miglitol"],
        "biguanide_sulfonylurea_gen2": ["glyburide-metformin", "glipizide-metformin"],
        "biguanide_thiazolidinedione": ["metformin-rosiglitazone", "metformin-pioglitazone"],
        "sulfonylurea_gen2_thiazolidinedione": ["glimepiride-pioglitazone"],
    }

    result = df.copy()
    priority = ["Up", "Down", "Steady"]

    for group, medications in groups.items():
        columns = df.columns.intersection(medications)
        if len(columns):
            values = df[columns]
            result = result.drop(columns=columns)
            result[group] = np.select([values.eq(status).any(axis=1) for status in priority], priority, default="No")

    return result

def engineer(df):
    visits = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]

    # Allow diminishing effects from additional previous visits.
    for column in visits:
        df[f"{column}_log1p"] = np.log1p(df[column])

    df["eng_prior_visits_log1p"] = np.log1p(df[visits].sum(axis=1))
    df["eng_prior_inpatient_any"] = (df["number_inpatient"] > 0).astype(int)
    df["eng_prior_inpatient_2plus"] = (df["number_inpatient"] >= 2).astype(int)

    # Let combined inpatient/emergency utilization have an additional effect.
    df["eng_inpatient_emergency_interaction"] = (
        df["number_inpatient_log1p"]
        * df["number_emergency_log1p"]
    )

    # Distinguish intensity from total volume during the admission.
    days = df["time_in_hospital"].clip(lower=1)
    df["eng_medications_per_day"] = df["num_medications"] / days
    df["eng_labs_per_day"] = df["num_lab_procedures"] / days

    return df


def features():
    chosen_features = [
    "number_inpatient",
    "number_emergency",
    "payer_code",
    "discharge_disposition_id",
    "diag_1",
    "number_diagnoses",
    "num_medications",
    ]
    return chosen_features


def final_engineered(Xt,Xv):

    """
    Pass in the training set and validation/test set. Drops the enconter_id and/or the readmitted column if there.
    Adds tons of features.
    
    Returns new Xt,Xv, and the list of features.
    """

    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    visits = ["number_outpatient", "number_emergency", "number_inpatient"]
    diags = ["diag_1", "diag_2", "diag_3"]
    drugs = Xt.loc[:, "metformin":"metformin-pioglitazone"].columns.tolist()

    frames = []
    for X in [Xt, Xv]:
        X.drop(columns=["encounter_id", "readmitted"], errors="ignore", inplace=True)
        grouped_diag = group_diagnoses(X)
        grouped_med = group_medications(X)
        for col in diags:
            X[col + "_group"] = grouped_diag[col]
        for col in grouped_med.columns.difference(X.columns):
            X[col] = grouped_med[col]
        engineer(X)
        X["eng_total_visits"] = X[visits].sum(axis=1, min_count=3)
        for col in visits:
            X[col + "_bin"] = pd.cut(X[col], [-1, 0, 1, 2, 4, np.inf],
                                    labels=["0", "1", "2", "3-4", "5+"]).astype(object)
            X[col + "_share"] = X[col] / (1 + X["eng_total_visits"])
        for col in numeric:
            X[col + "_log1p"] = np.log1p(X[col])
            X[col + "_squared"] = X[col] ** 2
        X = X.copy()
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                X[a + "_x_" + b] = X[a + "_log1p"] * X[b + "_log1p"]
        for col in ["num_procedures", "number_diagnoses"]:
            X[col + "_per_day"] = X[col] / X["time_in_hospital"].clip(lower=1)
        for status in ["No", "Steady", "Up", "Down"]:
            X["eng_drugs_" + status] = X[drugs].eq(status).sum(axis=1)
        X["eng_drugs_active"] = X[drugs].isin(["Steady", "Up", "Down"]).sum(axis=1)
        X["eng_drugs_changed"] = X[drugs].isin(["Up", "Down"]).sum(axis=1)
        X["eng_drugs_missing"] = X[drugs].isna().sum(axis=1)
        X["eng_age_midpoint"] = pd.to_numeric(
            X["age"].str.extract(r"\[(\d+)-", expand=False), errors="coerce") + 5
        for a, b in [("diag_1_group", "a1cresult"), ("diag_1_group", "insulin"),
                    ("age", "discharge_disposition_id"), ("insulin", "change"),
                    ("admission_type_id", "admission_source_id")]:
            X[a + "_with_" + b] = X[a].fillna("Missing").astype(str) + "|" + X[b].fillna("Missing").astype(str)
        for col in diags + ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            X[col] = X[col].astype(str).where(X[col].notna(), np.nan)
        for col in X.select_dtypes(exclude="number"):
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)

        frames.append(X)
    Xt, Xv = frames

    all_features = Xt.columns.tolist()



    return Xt, Xv, all_features