from django.shortcuts import render
import pandas as pd
import numpy as np
from . import ml_loader as ml
import gc
import time
from .mbti_utils import QUESTIONS, predict_mbti
import io
import base64
from PIL import Image
from django.shortcuts import render
from .plant_disease_data import IDX_TO_CLASS, DISEASE_ADVICE

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value, default=0.0):
    """Convert a POST string to float; return default if blank or invalid."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def all_filled(post, *field_names):
    """Return True only if every listed field is present and non-empty."""
    return all(post.get(f, '').strip() != '' for f in field_names)


# ── Home ──────────────────────────────────────────────────────────────────────

def home(request):
    return render(request, 'home.html')


# ── Heart Disease ─────────────────────────────────────────────────────────────

def heart_predict(request):
    result = None
    probability = None

    if request.method == 'POST':
        required = ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        if all_filled(request.POST, *required):
            age      = safe_float(request.POST.get('age'))
            sex      = safe_float(request.POST.get('sex'))
            cp       = safe_float(request.POST.get('cp'))
            trestbps = safe_float(request.POST.get('trestbps'))
            chol     = safe_float(request.POST.get('chol'))
            fbs      = safe_float(request.POST.get('fbs'))
            restecg  = safe_float(request.POST.get('restecg'))
            thalach  = safe_float(request.POST.get('thalach'))
            exang    = safe_float(request.POST.get('exang'))
            oldpeak  = safe_float(request.POST.get('oldpeak'))
            slope    = safe_float(request.POST.get('slope'))
            ca       = safe_float(request.POST.get('ca'))
            thal     = safe_float(request.POST.get('thal'))

            input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                       thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=ml.heart['columns'])
            scaled = ml.heart['scaler'].transform(input_df)
            prob   = ml.heart['model'].predict_proba(scaled)[0][1]
            result = "High Risk" if prob >= 0.5 else "Low Risk"
            probability = round(prob * 100, 1) if prob >= 0.5 else round((1 - prob) * 100, 1)

    return render(request, 'heart.html', {'result': result, 'probability': probability})


# ── Customer Churn ────────────────────────────────────────────────────────────

def churn_predict(request):
    result = None
    probability = None

    if request.method == 'POST':
        required = ('tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
                    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'gender',
                    'InternetService', 'Contract', 'PaymentMethod')
        if all_filled(request.POST, *required):
            tenure           = safe_float(request.POST.get('tenure'))
            monthly_charges  = safe_float(request.POST.get('MonthlyCharges'))
            total_charges    = safe_float(request.POST.get('TotalCharges'))
            senior           = safe_float(request.POST.get('SeniorCitizen'))
            partner          = safe_float(request.POST.get('Partner'))
            dependents       = safe_float(request.POST.get('Dependents'))
            phone_service    = safe_float(request.POST.get('PhoneService'))
            multiple_lines   = safe_float(request.POST.get('MultipleLines'))
            online_security  = safe_float(request.POST.get('OnlineSecurity'))
            online_backup    = safe_float(request.POST.get('OnlineBackup'))
            device_protect   = safe_float(request.POST.get('DeviceProtection'))
            tech_support     = safe_float(request.POST.get('TechSupport'))
            streaming_tv     = safe_float(request.POST.get('StreamingTV'))
            streaming_movies = safe_float(request.POST.get('StreamingMovies'))
            paperless        = safe_float(request.POST.get('PaperlessBilling'))
            gender           = safe_float(request.POST.get('gender'))
            internet_service = request.POST.get('InternetService')
            contract         = request.POST.get('Contract')
            payment_method   = request.POST.get('PaymentMethod')

            is_month_to_month = 1.0 if contract == 'Month-to-month' else 0.0
            avg_monthly_spend = total_charges / (tenure + 1)
            total_services    = (phone_service + multiple_lines + online_security +
                                 online_backup + device_protect + tech_support +
                                 streaming_tv + streaming_movies)

            input_df = pd.DataFrame([[gender, senior, partner, dependents, tenure,
                                       phone_service, multiple_lines, internet_service,
                                       online_security, online_backup, device_protect,
                                       tech_support, streaming_tv, streaming_movies,
                                       contract, paperless, payment_method,
                                       monthly_charges, total_charges,
                                       is_month_to_month, avg_monthly_spend, total_services]],
                                      columns=ml.churn['columns'])

            encoded = ml.churn['encoder'].transform(input_df)
            prob    = ml.churn['model'].predict_proba(encoded)[0][1]

            if prob >= ml.churn['threshold']:
                result      = "Likely to Churn"
                probability = round(prob * 100, 1)
            else:
                result      = "Likely to Stay"
                probability = round((1 - prob) * 100, 1)

    return render(request, 'churn.html', {'result': result, 'probability': probability})


# ── Employee Attrition ────────────────────────────────────────────────────────

def attrition_predict(request):
    result = None
    probability = None

    if request.method == 'POST':
        required = ('Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
                    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager')
        if all_filled(request.POST, *required):
            age                   = safe_float(request.POST.get('Age'))
            business_travel       = safe_float(request.POST.get('BusinessTravel'))
            daily_rate            = safe_float(request.POST.get('DailyRate'))
            department            = safe_float(request.POST.get('Department'))
            distance              = safe_float(request.POST.get('DistanceFromHome'))
            education             = safe_float(request.POST.get('Education'))
            education_field       = request.POST.get('EducationField')
            env_satisfaction      = safe_float(request.POST.get('EnvironmentSatisfaction'))
            gender                = safe_float(request.POST.get('Gender'))
            hourly_rate           = safe_float(request.POST.get('HourlyRate'))
            job_involvement       = safe_float(request.POST.get('JobInvolvement'))
            job_level             = safe_float(request.POST.get('JobLevel'))
            job_role              = request.POST.get('JobRole')
            job_satisfaction      = safe_float(request.POST.get('JobSatisfaction'))
            marital_status        = safe_float(request.POST.get('MaritalStatus'))
            monthly_income        = safe_float(request.POST.get('MonthlyIncome'))
            monthly_rate          = safe_float(request.POST.get('MonthlyRate'))
            num_companies         = safe_float(request.POST.get('NumCompaniesWorked'))
            overtime              = safe_float(request.POST.get('OverTime'))
            pct_hike              = safe_float(request.POST.get('PercentSalaryHike'))
            perf_rating           = safe_float(request.POST.get('PerformanceRating'))
            rel_satisfaction      = safe_float(request.POST.get('RelationshipSatisfaction'))
            stock_option          = safe_float(request.POST.get('StockOptionLevel'))
            total_working_years   = safe_float(request.POST.get('TotalWorkingYears'))
            training_times        = safe_float(request.POST.get('TrainingTimesLastYear'))
            work_life_balance     = safe_float(request.POST.get('WorkLifeBalance'))
            years_at_company      = safe_float(request.POST.get('YearsAtCompany'))
            years_in_role         = safe_float(request.POST.get('YearsInCurrentRole'))
            years_since_promotion = safe_float(request.POST.get('YearsSinceLastPromotion'))
            years_with_manager    = safe_float(request.POST.get('YearsWithCurrManager'))

            avg_satisfaction   = np.mean([env_satisfaction, job_satisfaction, rel_satisfaction, work_life_balance])
            income_per_exp     = monthly_income / (total_working_years + 1)
            stagnation_ratio   = years_since_promotion / (years_at_company + 1)
            is_loyal           = 1.0 if years_at_company > 5 else 0.0
            is_overworked      = 1.0 if (overtime == 1 and work_life_balance <= 2) else 0.0
            career_growth_rate = job_level / (total_working_years + 1)

            input_df = pd.DataFrame([[age, business_travel, daily_rate, department,
                                       distance, education, education_field,
                                       env_satisfaction, gender, hourly_rate,
                                       job_involvement, job_level, job_role,
                                       job_satisfaction, marital_status, monthly_income,
                                       monthly_rate, num_companies, overtime, pct_hike,
                                       perf_rating, rel_satisfaction, stock_option,
                                       total_working_years, training_times, work_life_balance,
                                       years_at_company, years_in_role, years_since_promotion,
                                       years_with_manager, avg_satisfaction, income_per_exp,
                                       stagnation_ratio, is_loyal, is_overworked, career_growth_rate]],
                                      columns=ml.attrition['columns'])

            encoded = ml.attrition['encoder'].transform(input_df)
            prob    = ml.attrition['model'].predict_proba(encoded)[0][1]

            if prob >= ml.attrition['threshold']:
                result      = "Likely to Leave"
                probability = round(prob * 100, 1)
            else:
                result      = "Likely to Stay"
                probability = round((1 - prob) * 100, 1)

    return render(request, 'attrition.html', {'result': result, 'probability': probability})


# ── Adult Census Income ───────────────────────────────────────────────────────

def census_predict(request):
    result = None
    probability = None
    post_vals = {}

    if request.method == 'POST':
        post_vals = {
            'age':           request.POST.get('age', ''),
            'education_num': request.POST.get('education.num', ''),
            'sex':           request.POST.get('sex', ''),
            'marital':       request.POST.get('marital.status', ''),
            'workclass':     request.POST.get('workclass', ''),
            'occupation':    request.POST.get('occupation', ''),
            'relationship':  request.POST.get('relationship', ''),
            'native':        request.POST.get('native.country', ''),
            'capital_gain':  request.POST.get('capital.gain', ''),
            'capital_loss':  request.POST.get('capital.loss', ''),
            'hours':         request.POST.get('hours.per.week', ''),
        }

        required = ('age', 'education.num', 'sex', 'marital.status', 'workclass',
                    'occupation', 'relationship', 'native.country',
                    'capital.gain', 'capital.loss', 'hours.per.week')
        if all_filled(request.POST, *required):
            age            = safe_float(request.POST.get('age'))
            education_num  = safe_float(request.POST.get('education.num'), 9.0)
            sex            = safe_float(request.POST.get('sex'))
            marital        = safe_float(request.POST.get('marital.status'))
            workclass      = request.POST.get('workclass')
            occupation     = request.POST.get('occupation')
            relationship   = request.POST.get('relationship')
            native_region  = request.POST.get('native.country')
            capital_gain   = safe_float(request.POST.get('capital.gain'))
            capital_loss   = safe_float(request.POST.get('capital.loss'))
            hours_per_week = safe_float(request.POST.get('hours.per.week'), 40.0)

            fnlwgt = 189778.0
            race   = 'White'

            age_edu     = age * education_num
            has_capital = 1 if (capital_gain > 0 or capital_loss > 0) else 0

            input_df = pd.DataFrame([[age, workclass, fnlwgt, education_num,
                                       marital, occupation, relationship, race,
                                       sex, capital_gain, capital_loss,
                                       hours_per_week, native_region,
                                       age_edu, has_capital]],
                                      columns=ml.census['columns'])

            encoded = ml.census['encoder'].transform(input_df)
            prob    = ml.census['model'].predict_proba(encoded)[0][1]

            if prob >= ml.census['threshold']:
                result      = 'Income > $50K'
                probability = round(prob * 100, 1)
            else:
                result      = 'Income <= $50K'
                probability = round((1 - prob) * 100, 1)

    return render(request, 'census.html', {
        'result':      result,
        'probability': probability,
        'pv':          post_vals,
    })


# ── Ames Housing Price ────────────────────────────────────────────────────────

def ames_predict(request):
    predicted_price = None
    post_data = {}

    if request.method == 'POST':
        post_data = request.POST

        required = ('Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
                    'Lot Area', 'Lot Frontage', 'Lot Shape', 'Neighborhood',
                    'MS Zoning', 'Bldg Type', 'House Style', 'Exter Qual',
                    'Kitchen Qual', 'Bsmt Qual', 'Bsmt Exposure',
                    'Gr Liv Area', '1st Flr SF', '2nd Flr SF', 'Total Bsmt SF',
                    'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath',
                    'Bedroom AbvGr', 'Kitchen AbvGr', 'Garage Cars', 'Garage Area',
                    'Garage Type', 'Sale Condition')
        if all_filled(request.POST, *required):

            # Ordinal maps — exact category order from the notebook's OrdinalEncoder
            LOT_SHAPE     = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
            LAND_SLOPE    = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
            QUAL5         = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
            BSMT_QUAL     = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            BSMT_EXP      = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
            BSMTFIN       = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
            FUNCTIONAL    = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
            GARAGE_FINISH = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
            GARAGE_QUAL   = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            PAVED_DRIVE   = {'N': 0, 'P': 1, 'Y': 2}

            # Raw POST values
            overall_qual   = safe_float(request.POST.get('Overall Qual'))
            overall_cond   = safe_float(request.POST.get('Overall Cond'))
            year_built     = safe_float(request.POST.get('Year Built'))
            year_remod     = safe_float(request.POST.get('Year Remod/Add'))
            lot_area       = safe_float(request.POST.get('Lot Area'))
            lot_frontage   = safe_float(request.POST.get('Lot Frontage'))
            gr_liv_area    = safe_float(request.POST.get('Gr Liv Area'))
            first_flr_sf   = safe_float(request.POST.get('1st Flr SF'))
            second_flr_sf  = safe_float(request.POST.get('2nd Flr SF'))
            total_bsmt_sf  = safe_float(request.POST.get('Total Bsmt SF'))
            full_bath      = safe_float(request.POST.get('Full Bath'))
            half_bath      = safe_float(request.POST.get('Half Bath'))
            bsmt_full_bath = safe_float(request.POST.get('Bsmt Full Bath'))
            bsmt_half_bath = safe_float(request.POST.get('Bsmt Half Bath'))
            bedroom_abvgr  = safe_float(request.POST.get('Bedroom AbvGr'))
            kitchen_abvgr  = safe_float(request.POST.get('Kitchen AbvGr'))
            garage_cars    = safe_float(request.POST.get('Garage Cars'))
            garage_area    = safe_float(request.POST.get('Garage Area'))

            # Nominal (string) columns - passed to OneHotEncoder inside preprocessor
            ms_zoning      = request.POST.get('MS Zoning')
            neighborhood   = request.POST.get('Neighborhood')
            bldg_type      = request.POST.get('Bldg Type')
            house_style    = request.POST.get('House Style')
            garage_type    = request.POST.get('Garage Type')
            sale_condition = request.POST.get('Sale Condition')
            sale_type      = request.POST.get('Sale Type', 'WD')

            # Ordinal columns - must be encoded to numbers exactly as the notebook did
            lot_shape      = LOT_SHAPE.get(request.POST.get('Lot Shape', ''), 3)
            exter_qual     = QUAL5.get(request.POST.get('Exter Qual', ''), 2)
            exter_cond     = QUAL5.get('TA', 2)           # hardcoded default
            bsmt_qual      = BSMT_QUAL.get(request.POST.get('Bsmt Qual', ''), 3)
            bsmt_cond      = BSMT_QUAL.get('TA', 3)       # hardcoded default
            bsmt_exposure  = BSMT_EXP.get(request.POST.get('Bsmt Exposure', ''), 0)
            kitchen_qual   = QUAL5.get(request.POST.get('Kitchen Qual', ''), 2)
            heating_qc     = QUAL5.get('Ex', 4)           # hardcoded default
            functional     = FUNCTIONAL.get('Typ', 7)     # hardcoded default
            garage_finish  = GARAGE_FINISH.get('Unf', 1)  # hardcoded default
            garage_qual    = GARAGE_QUAL.get('TA', 3)     # hardcoded default
            garage_cond    = GARAGE_QUAL.get('TA', 3)     # hardcoded default
            paved_drive    = PAVED_DRIVE.get('Y', 2)      # hardcoded default
            land_slope     = LAND_SLOPE.get('Gtl', 2)     # hardcoded default
            bsmtfin_type1  = BSMTFIN.get('Unf', 1)        # hardcoded default
            bsmtfin_type2  = BSMTFIN.get('Unf', 1)        # hardcoded default

            # Hardcoded constants (nominal - strings, go to OneHotEncoder)
            ms_subclass     = 20.0
            street          = request.POST.get('Street', 'Pave')
            land_contour    = 'Lvl'
            utilities       = request.POST.get('Utilities', 'AllPub')
            lot_config      = 'Inside'
            condition_1     = 'Norm'
            condition_2     = request.POST.get('Condition 2', 'Norm')
            roof_style      = 'Gable'
            roof_matl       = request.POST.get('Roof Matl', 'CompShg')
            exterior_1st    = 'VinylSd'
            exterior_2nd    = 'VinylSd'
            foundation      = 'PConc'
            heating         = request.POST.get('Heating', 'GasA')
            central_air     = 'Y'
            electrical      = 'SBrkr'

            # Hardcoded constants (numeric)
            yr_sold         = 2010.0
            mas_vnr_area    = 0.0
            bsmtfin_sf1     = 0.0
            bsmtfin_sf2     = 0.0
            bsmt_unf_sf     = total_bsmt_sf
            low_qual_fin_sf = safe_float(request.POST.get('Low Qual Fin SF', '0'))
            totrms_abvgrd   = bedroom_abvgr + kitchen_abvgr + 1
            fireplaces      = 0.0
            garage_yr_blt   = year_built
            wood_deck_sf    = 0.0
            open_porch_sf   = 0.0
            enclosed_porch  = 0.0
            three_ssn_porch = safe_float(request.POST.get('3Ssn Porch', '0'))
            screen_porch    = safe_float(request.POST.get('Screen Porch', '0'))
            pool_area       = safe_float(request.POST.get('Pool Area', '0'))
            misc_val        = safe_float(request.POST.get('Misc Val', '0'))
            mo_sold         = 6.0

            # Feature engineering
            total_sf        = total_bsmt_sf + first_flr_sf + second_flr_sf
            total_bathrooms = full_bath + half_bath * 0.5 + bsmt_full_bath + bsmt_half_bath * 0.5
            house_age       = yr_sold - year_built
            remod_age       = yr_sold - year_remod

            input_df = pd.DataFrame([[
                ms_subclass, ms_zoning, lot_frontage, lot_area, street,
                lot_shape, land_contour, utilities, lot_config, land_slope,
                neighborhood, condition_1, condition_2, bldg_type, house_style,
                overall_qual, overall_cond, year_built, year_remod,
                roof_style, roof_matl, exterior_1st, exterior_2nd,
                mas_vnr_area, exter_qual, exter_cond, foundation,
                bsmt_qual, bsmt_cond, bsmt_exposure,
                bsmtfin_type1, bsmtfin_sf1, bsmtfin_type2, bsmtfin_sf2,
                bsmt_unf_sf, total_bsmt_sf,
                heating, heating_qc, central_air, electrical,
                first_flr_sf, second_flr_sf, low_qual_fin_sf, gr_liv_area,
                bsmt_full_bath, bsmt_half_bath, full_bath, half_bath,
                bedroom_abvgr, kitchen_abvgr, kitchen_qual, totrms_abvgrd,
                functional, fireplaces,
                garage_type, garage_yr_blt, garage_finish,
                garage_cars, garage_area, garage_qual, garage_cond,
                paved_drive,
                wood_deck_sf, open_porch_sf, enclosed_porch,
                three_ssn_porch, screen_porch, pool_area, misc_val,
                mo_sold, yr_sold, sale_type, sale_condition,
                total_sf, total_bathrooms, house_age, remod_age
            ]], columns=ml.ames['columns'])

            transformed     = ml.ames['preprocessor'].transform(input_df)
            log_pred        = ml.ames['model'].predict(transformed)[0]
            price           = np.expm1(log_pred)
            predicted_price = f"{price:,.0f}"

    return render(request, 'ames.html', {
        'predicted_price': predicted_price,
        'f_ms_zoning':      request.POST.get('MS Zoning', ''),
        'f_neighborhood':   request.POST.get('Neighborhood', ''),
        'f_bldg_type':      request.POST.get('Bldg Type', ''),
        'f_house_style':    request.POST.get('House Style', ''),
        'f_lot_shape':      request.POST.get('Lot Shape', ''),
        'f_sale_condition': request.POST.get('Sale Condition', ''),
        'f_overall_qual':   request.POST.get('Overall Qual', ''),
        'f_overall_cond':   request.POST.get('Overall Cond', ''),
        'f_exter_qual':     request.POST.get('Exter Qual', ''),
        'f_kitchen_qual':   request.POST.get('Kitchen Qual', ''),
        'f_bsmt_qual':      request.POST.get('Bsmt Qual', ''),
        'f_bsmt_exposure':  request.POST.get('Bsmt Exposure', ''),
        'f_garage_type':    request.POST.get('Garage Type', ''),
        'f_year_built':     request.POST.get('Year Built', ''),
        'f_year_remod':     request.POST.get('Year Remod/Add', ''),
        'f_lot_area':       request.POST.get('Lot Area', ''),
        'f_gr_liv_area':    request.POST.get('Gr Liv Area', ''),
        'f_1st_flr_sf':     request.POST.get('1st Flr SF', ''),
        'f_2nd_flr_sf':     request.POST.get('2nd Flr SF', ''),
        'f_total_bsmt_sf':  request.POST.get('Total Bsmt SF', ''),
        'f_bsmt_full_bath': request.POST.get('Bsmt Full Bath', ''),
        'f_bsmt_half_bath': request.POST.get('Bsmt Half Bath', ''),
        'f_full_bath':      request.POST.get('Full Bath', ''),
        'f_half_bath':      request.POST.get('Half Bath', ''),
        'f_bedroom_abvgr':  request.POST.get('Bedroom AbvGr', ''),
        'f_kitchen_abvgr':  request.POST.get('Kitchen AbvGr', ''),
        'f_lot_frontage':   request.POST.get('Lot Frontage', ''),
        'f_garage_cars':    request.POST.get('Garage Cars', ''),
        'f_garage_area':    request.POST.get('Garage Area', ''),
        'post_data':        post_data,
    })

# ── Student Performance Prediction ────────────────────────────────────────────────────────

def student_predict(request):
    predicted_score = None

    if request.method == 'POST':
        required = (
            'hours_studied', 'previous_scores', 'extracurricular',
            'sleep_hours', 'sample_papers',
        )
        if all_filled(request.POST, *required):
            hours_studied   = safe_float(request.POST.get('hours_studied'))
            previous_scores = safe_float(request.POST.get('previous_scores'))
            extracurricular = request.POST.get('extracurricular')   # '0' or '1'
            sleep_hours     = safe_float(request.POST.get('sleep_hours'))
            sample_papers   = safe_float(request.POST.get('sample_papers'))

            # Notebook mapped: Yes -> 0, No -> 1 before fitting
            extracurricular_encoded = int(extracurricular)

            m = ml.student
            df = pd.DataFrame([{
                'Hours Studied':                   hours_studied,
                'Previous Scores':                 previous_scores,
                'Extracurricular Activities':      extracurricular_encoded,
                'Sleep Hours':                     sleep_hours,
                'Sample Question Papers Practiced': sample_papers,
            }], columns=m['columns'])

            pred = m['model'].predict(df)[0]
            predicted_score = round(float(pred), 1)

    return render(request, 'student.html', {'predicted_score': predicted_score})

# ── Customer Segmentation ─────────────────────────────────────────────────────

from datetime import date

def segmentation_predict(request):
    segment = None
    description = None
    submitted = False
    today = date.today()

    if request.method == 'POST':
        submitted = True
        if all_filled(request.POST, 'last_order_date', 'total_spend'):
            try:
                last_order_str  = request.POST.get('last_order_date')
                total_spend     = safe_float(request.POST.get('total_spend'))
                last_order_date = date.fromisoformat(last_order_str)
                recency         = (today - last_order_date).days

                recency_log  = np.log1p(recency)
                monetary_log = np.log1p(total_spend)

                input_scaled = ml.segmentation['scaler'].transform([[recency_log, monetary_log]])
                cluster_id   = str(ml.segmentation['model'].predict(input_scaled)[0])
                segment      = ml.segmentation['labels'][cluster_id]

                descriptions = {
                    'Active Customers':  'Purchased recently. High engagement, good candidate for upsell.',
                    'Lapsed High-Value': 'Once a high spender, now inactive. Strong win-back candidate.',
                    'Lost Cheap':        'Low spend, long inactive. Low priority for reactivation.',
                }
                description = descriptions.get(segment, '')

            except Exception as e:
                print("SEGMENTATION ERROR:", e)

    return render(request, 'segmentation.html', {
        'segment':     segment,
        'description': description,
        'submitted':   submitted,
        'today':       today,
    })

def privacy(request):
    return render(request, 'privacy.html')

def terms(request):
    return render(request, 'terms.html')

# ── MBTI ─────────────────────────────────────────────────────

def mbti_predict(request):
    result = None
    confidences = None

    if request.method == 'POST':
        answers = {}
        for question in QUESTIONS:
            answers[question['id']] = float(request.POST[question['id']])

        result, confidences = predict_mbti(answers, ml.mbti)

    return render(request, 'mbti.html', {
        'questions': QUESTIONS,
        'result': result,
        'confidences': confidences,
    })

# ── Plant Disease ─────────────────────────────────────────────────────

import io
import base64
from PIL import Image
from django.shortcuts import render

from .plant_disease_data import IDX_TO_CLASS, DISEASE_ADVICE


def preprocess_image(image: Image.Image, size: int = 256):
    image = image.convert('RGB')
    original_size = image.size

    image_resized = image.resize((size, size), Image.BILINEAR)
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = image_array.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

    return image_tensor, original_size

MAX_DIM = 512  # longest side, in pixels, before any processing touches the image

def _load_and_cap_image(uploaded_file, max_dim: int = MAX_DIM) -> Image.Image:
    """
    Uses JPEG draft mode to decode directly at a reduced resolution where
    possible, avoiding ever allocating the full native-resolution array.
    Falls back to a plain resize for non-JPEG uploads or if draft mode
    doesn't shrink it enough on its own.
    """
    image = Image.open(uploaded_file)
    try:
        image.draft('RGB', (max_dim, max_dim))
    except Exception:
        pass
    image = image.convert('RGB')

    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    return image

def looks_like_plant_leaf(image: Image.Image, min_green_fraction: float = 0.12) -> bool:
    img = image.convert('RGB').resize((128, 128))
    arr = np.array(img).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (g > r) & (g > b) & (g > 40)
    green_fraction = np.sum(green_mask) / green_mask.size
    return green_fraction >= min_green_fraction

def predict_mask(image: Image.Image, session):
    image_tensor, original_size = preprocess_image(image)

    output = session.run(['output'], {'input': image_tensor})[0]  # (1, 16, 256, 256)
    pred_mask = np.argmax(output[0], axis=0).astype(np.uint8)     # (256, 256)

    pred_mask_img = Image.fromarray(pred_mask)
    pred_mask_resized = pred_mask_img.resize(original_size, Image.NEAREST)
    pred_mask_full = np.array(pred_mask_resized)

    unique_classes, counts = np.unique(pred_mask_full, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    class_counts.pop(0, None)  # drop background

    if not class_counts:
        return pred_mask_full, "No disease detected", 0

    predicted_class_idx = max(class_counts, key=class_counts.get)  # channel idx, 1-15
    predicted_disease = IDX_TO_CLASS[predicted_class_idx - 1]      # shift to 0-14
    confidence_pixels = int(class_counts[predicted_class_idx])

    return pred_mask_full, predicted_disease, confidence_pixels


def create_overlay(original_image: Image.Image, pred_mask: np.ndarray, alpha: float = 0.5):
    original = original_image.convert('RGB')
    original_arr = np.array(original).astype(np.float32)

    red_layer = np.zeros_like(original_arr)
    red_layer[..., 0] = 255

    disease_area = (pred_mask > 0)[..., np.newaxis]
    blended = np.where(
        disease_area,
        original_arr * (1 - alpha) + red_layer * alpha,
        original_arr
    ).astype(np.uint8)

    return Image.fromarray(blended)


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plant_disease_predict(request):
    result = None
    advice = None
    overlay_b64 = None
    pixel_count = None

    if request.method == 'POST' and request.FILES.get('image'):
        if ml.plant_disease_session is None:
            return render(request, 'plant_disease.html', {
                'error': 'Model failed to load on the server. Check server logs.'
            })

        uploaded = request.FILES['image']
        t0 = time.time()
        try:
            original_image = _load_and_cap_image(uploaded)
        except Exception:
            return render(request, 'plant_disease.html', {
                'error': 'Could not read that file as an image.'
            })
        t1 = time.time()

        if not looks_like_plant_leaf(original_image):
            return render(request, 'plant_disease.html', {
                'error': 'This doesn\'t look like a plant leaf photo. Please upload a clear, well-lit photo of a single leaf.'
            })

        pred_mask, predicted_disease, pixel_count = predict_mask(original_image, ml.plant_disease_session)
        t2 = time.time()

        result = predicted_disease
        if predicted_disease != "No disease detected":
            advice = DISEASE_ADVICE.get(predicted_disease)
            overlay_img = create_overlay(original_image, pred_mask)
            overlay_b64 = _image_to_base64(overlay_img)
            del overlay_img
        t3 = time.time()

        print(f"[plant-disease timing] decode+cap: {t1-t0:.2f}s | inference: {t2-t1:.2f}s | overlay+encode: {t3-t2:.2f}s | total: {t3-t0:.2f}s")

        del original_image, pred_mask
        gc.collect()

    return render(request, 'plant_disease.html', {
        'result': result,
        'advice': advice,
        'overlay_b64': overlay_b64,
        'pixel_count': pixel_count,
    })