
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import xgboost as xgb
import numpy as np
import json
import csv
import io
import math
import os
from datetime import datetime


def parse_datetime_value(value):
    """Accept common ISO-like and legacy timestamp strings."""

    if value is None:
        raise ValueError("Date and time value is required.")

    candidate = str(value).strip()
    if not candidate:
        raise ValueError("Date and time value is required.")

    normalized = candidate.replace("Z", "+00:00")
    iso_formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    )

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in iso_formats:
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue

    raise ValueError(
        "Invalid date_time format. Use YYYY-MM-DD HH:MM:SS or ISO-8601."
    )


def validate_prediction_request(payload):
    """Support both Pydantic v1 and v2 validation APIs."""

    if hasattr(PredictionRequest, "model_validate"):
        return PredictionRequest.model_validate(payload)
    return PredictionRequest(**payload)


def copy_prediction_request(request, **updates):
    """Support both Pydantic v1 and v2 copy/update APIs."""

    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    copied = request.copy()
    copied.update(updates)
    return copied


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "renewable_energy_xgboost_model.json"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "source_key_mapping.json"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "final_metrics.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.Booster()
model.load_model(MODEL_PATH)


# ============================================================
# LOAD FEATURES
# ============================================================

with open(FEATURE_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)


# ============================================================
# LOAD SOURCE KEY MAPPING
# ============================================================

with open(MAPPING_PATH, "r") as f:
    SOURCE_KEY_MAPPING = json.load(f)


# ============================================================
# LOAD METRICS
# ============================================================

with open(METRICS_PATH, "r") as f:
    MODEL_METRICS = json.load(f)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Renewable Energy Output Prediction API",
    description=(
        "API for predicting solar AC power output "
        "using a trained XGBoost regression model."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    plant_id: int = Field(
        ...,
        description="Plant ID"
    )

    ambient_temperature: float = Field(
        ...,
        description="Ambient temperature in Celsius"
    )

    module_temperature: float = Field(
        ...,
        description="Module temperature in Celsius"
    )

    irradiation: float = Field(
        ...,
        ge=0,
        description="Solar irradiation"
    )

    source_key: str = Field(
        ...,
        description="Inverter/source key"
    )

    date_time: str = Field(
        ...,
        description="Date and time in YYYY-MM-DD HH:MM:SS format"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "project": "Renewable Energy Output Prediction",
        "model": "Tuned XGBoost",
        "target": "AC_POWER",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "features": len(FEATURE_NAMES)
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": MODEL_METRICS["model"],
        "target": MODEL_METRICS["target"],
        "test_samples": MODEL_METRICS["test_samples"],
        "mae": MODEL_METRICS["MAE"],
        "rmse": MODEL_METRICS["RMSE"],
        "r2": MODEL_METRICS["R2"],
        "number_of_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Parse date/time
        # ----------------------------------------------------

        try:
            dt = parse_datetime_value(request.date_time)

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error)
            ) from error


        # ----------------------------------------------------
        # Validate source key
        # ----------------------------------------------------

        source_key = str(request.source_key)

        if source_key not in SOURCE_KEY_MAPPING:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown source_key: {source_key}. "
                    "Please use a valid inverter/source key."
                )
            )


        # ----------------------------------------------------
        # Validate plant
        # ----------------------------------------------------

        valid_plants = [4135001, 4136001]

        if request.plant_id not in valid_plants:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid plant_id. "
                    "Expected 4135001 or 4136001."
                )
            )


        # ----------------------------------------------------
        # Create time features
        # ----------------------------------------------------

        year = dt.year
        month = dt.month
        day = dt.day
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        minute = dt.minute

        time_in_hours = (
            hour +
            minute / 60.0
        )


        # ----------------------------------------------------
        # Cyclical features
        # ----------------------------------------------------

        hour_sin = math.sin(
            2 * math.pi * hour / 24
        )

        hour_cos = math.cos(
            2 * math.pi * hour / 24
        )

        day_of_year_sin = math.sin(
            2 * math.pi * day_of_year / 365
        )

        day_of_year_cos = math.cos(
            2 * math.pi * day_of_year / 365
        )

        day_of_week_sin = math.sin(
            2 * math.pi * day_of_week / 7
        )

        day_of_week_cos = math.cos(
            2 * math.pi * day_of_week / 7
        )


        # ----------------------------------------------------
        # Engineered weather features
        # ----------------------------------------------------

        temp_difference = (
            request.module_temperature -
            request.ambient_temperature
        )

        irradiation_squared = (
            request.irradiation ** 2
        )

        module_temp_squared = (
            request.module_temperature ** 2
        )


        # ----------------------------------------------------
        # SOURCE KEY ENCODING
        # ----------------------------------------------------

        source_key_encoded = SOURCE_KEY_MAPPING[source_key]


        # ----------------------------------------------------
        # BUILD FEATURE VECTOR
        # ----------------------------------------------------

        feature_values = {

            "PLANT_ID": request.plant_id,

            "AMBIENT_TEMPERATURE":
                request.ambient_temperature,

            "MODULE_TEMPERATURE":
                request.module_temperature,

            "IRRADIATION":
                request.irradiation,

            "YEAR": year,
            "MONTH": month,
            "DAY": day,
            "DAY_OF_WEEK": day_of_week,
            "DAY_OF_YEAR": day_of_year,
            "HOUR": hour,
            "MINUTE": minute,

            "TIME_IN_HOURS":
                time_in_hours,

            "HOUR_SIN":
                hour_sin,

            "HOUR_COS":
                hour_cos,

            "DAY_OF_YEAR_SIN":
                day_of_year_sin,

            "DAY_OF_YEAR_COS":
                day_of_year_cos,

            "DAY_OF_WEEK_SIN":
                day_of_week_sin,

            "DAY_OF_WEEK_COS":
                day_of_week_cos,

            "TEMP_DIFFERENCE":
                temp_difference,

            "IRRADIATION_SQUARED":
                irradiation_squared,

            "MODULE_TEMP_SQUARED":
                module_temp_squared,

            "SOURCE_KEY_ENCODED":
                source_key_encoded
        }


        # ----------------------------------------------------
        # ENSURE EXACT FEATURE ORDER
        # ----------------------------------------------------

        input_data = np.array(
            [[feature_values[f] for f in FEATURE_NAMES]],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        dmatrix = xgb.DMatrix(
            input_data,
            feature_names=FEATURE_NAMES
        )

        raw_prediction = float(
            model.predict(dmatrix)[0]
        )

        contribution_values = model.predict(
            dmatrix,
            pred_contribs=True
        )[0]

        feature_contributions = [
            {
                "feature": feature_name,
                "contribution": round(float(contribution), 4)
            }
            for feature_name, contribution in zip(
                FEATURE_NAMES,
                contribution_values[:-1]
            )
        ]


        # ----------------------------------------------------
        # PHYSICAL CONSTRAINT
        # ----------------------------------------------------
        # AC power cannot be negative.

        prediction = max(
            0.0,
            raw_prediction
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": round(
                prediction,
                4
            ),

            "unit": "kW",

            "raw_prediction": round(
                raw_prediction,
                4
            ),

            "plant_id": request.plant_id,

            "source_key": source_key,

            "date_time": request.date_time,

            "model": "Tuned XGBoost",

            "feature_contributions": feature_contributions,

            "base_value": round(float(contribution_values[-1]), 4)
        }


    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/predict/day-profile")
def predict_day_profile(request: PredictionRequest):

    try:
        selected_datetime = parse_datetime_value(request.date_time)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    profile = []
    selected_sunlight = max(
        0.0,
        math.sin(math.pi * (selected_datetime.hour - 6) / 12)
    )
    peak_irradiation = (
        min(1.0, request.irradiation / selected_sunlight)
        if selected_sunlight > 0.1
        else request.irradiation
    )

    for hour in range(24):
        sunlight_level = max(
            0.0,
            math.sin(math.pi * (hour - 6) / 12)
        )
        hourly_irradiation = peak_irradiation * sunlight_level
        temperature_fraction = (
            min(1.0, hourly_irradiation / request.irradiation)
            if request.irradiation > 0
            else sunlight_level
        )
        hourly_request = copy_prediction_request(
            request,
            date_time=selected_datetime.replace(
                hour=hour,
                minute=0,
                second=0
            ).strftime("%Y-%m-%d %H:%M:%S"),
            irradiation=hourly_irradiation,
            module_temperature=request.ambient_temperature + (
                request.module_temperature - request.ambient_temperature
            ) * temperature_fraction
        )
        prediction = predict(hourly_request)
        profile.append({
            "hour": hour,
            "prediction": prediction["prediction"],
            "irradiation": round(hourly_irradiation, 4)
        })

    return {"date": selected_datetime.date().isoformat(), "profile": profile}


@app.post("/predict/batch")
async def predict_batch(request: Request):

    csv_bytes = await request.body()

    if len(csv_bytes) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="CSV files must be 10 MB or smaller."
        )

    try:
        csv_text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="CSV must use UTF-8 encoding."
        ) from error

    reader = csv.DictReader(io.StringIO(csv_text))
    required_columns = {
        "plant_id",
        "ambient_temperature",
        "module_temperature",
        "irradiation",
        "source_key",
        "date_time"
    }
    supplied_columns = set(reader.fieldnames or [])
    missing_columns = sorted(required_columns - supplied_columns)

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail="Missing CSV columns: " + ", ".join(missing_columns)
        )

    results = []

    for row_number, row in enumerate(reader, start=2):
        if len(results) >= 5000:
            raise HTTPException(
                status_code=413,
                detail="CSV files may contain at most 5,000 data rows."
            )

        row["date_time"] = row["date_time"].strip()
        if len(row["date_time"]) == 16 and "T" in row["date_time"]:
            row["date_time"] += ":00"

        try:
            prediction_request = validate_prediction_request(row)
            result = predict(prediction_request)
        except HTTPException as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=f"CSV row {row_number}: {error.detail}"
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"CSV row {row_number}: {error}"
            ) from error

        result.update({
            "ambient_temperature": prediction_request.ambient_temperature,
            "module_temperature": prediction_request.module_temperature,
            "irradiation": prediction_request.irradiation
        })
        result["row_number"] = row_number
        results.append(result)

    if not results:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows."
        )

    return {"count": len(results), "results": results}

# ============================================================
# FRONTEND
# ============================================================

FRONTEND_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "renewable_energy_frontend")
)

if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/app",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend"
    )
