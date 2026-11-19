#!/usr/bin/env python3
"""
SOLIS Batch Analysis - Extract real-world statistics for paper revision.

This script:
1. Extracts SoLEXS and HEL1OS FITS files from nested ZIPs
2. Analyzes light curve data to identify flare signatures
3. Computes aggregate statistics with confidence intervals
4. Outputs results as JSON for paper incorporation

Usage:
    python batch_analysis.py
"""
import os
import sys
import json
import zipfile
import glob
import traceback
import numpy as np
from pathlib import Path

# Add the Python source code directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = SCRIPT_DIR
sys.path.insert(0, os.path.abspath(PYTHON_DIR))

from fits_reader import read_fits_file

SOLEXS_ZIP_DIR = os.path.join(SCRIPT_DIR, 'datasets', 'solexs')
HEL1OS_ZIP_DIR = os.path.join(SCRIPT_DIR, 'datasets', 'hel1os')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_results')
EXTRACT_DIR = os.path.join(SCRIPT_DIR, 'datasets', '_extracted')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


def extract_solexs_fits(zip_path):
    """Extract .lc files from a SoLEXS ZIP (may contain nested ZIPs)."""
    extracted_files = []
    date_str = os.path.basename(zip_path).replace('AL1_SLX_L1_', '').replace('_v1.0.zip', '')
    extract_subdir = os.path.join(EXTRACT_DIR, 'solexs', date_str)
    os.makedirs(extract_subdir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as outer:
            outer.extractall(extract_subdir)

        # Search for .lc files
        lc_files = glob.glob(os.path.join(extract_subdir, '**', '*.lc'), recursive=True)
        
        # Also look for inner ZIPs
        inner_zips = glob.glob(os.path.join(extract_subdir, '**', '*.zip'), recursive=True)
        for iz in inner_zips:
            try:
                with zipfile.ZipFile(iz, 'r') as inner:
                    inner.extractall(os.path.dirname(iz))
                lc_files.extend(glob.glob(os.path.join(os.path.dirname(iz), '**', '*.lc'), recursive=True))
            except:
                pass

        extracted_files = list(set(lc_files))
    except Exception as e:
        print(f"  ERROR extracting {zip_path}: {e}")

    return extracted_files


def extract_hel1os_fits(zip_path):
    """Extract FITS/lc files from a HEL1OS ZIP."""
    extracted_files = []
    basename = os.path.splitext(os.path.basename(zip_path))[0]
    extract_subdir = os.path.join(EXTRACT_DIR, 'hel1os', basename)
    os.makedirs(extract_subdir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as outer:
            outer.extractall(extract_subdir)

        for ext in ['*.lc', '*.fits', '*.fit', '*.pi']:
            extracted_files.extend(
                glob.glob(os.path.join(extract_subdir, '**', ext), recursive=True)
            )
        
        inner_zips = glob.glob(os.path.join(extract_subdir, '**', '*.zip'), recursive=True)
        for iz in inner_zips:
            try:
                with zipfile.ZipFile(iz, 'r') as inner:
                    inner.extractall(os.path.dirname(iz))
                for ext in ['*.lc', '*.fits', '*.fit', '*.pi']:
                    extracted_files.extend(
                        glob.glob(os.path.join(os.path.dirname(iz), '**', ext), recursive=True)
                    )
            except:
                pass

        extracted_files = list(set(extracted_files))
    except Exception as e:
        print(f"  ERROR extracting {zip_path}: {e}")

    return extracted_files


def analyze_solexs_file(lc_path):
    """Analyze a single SoLEXS .lc file. Returns statistics dict."""
    result = {
        'file': os.path.basename(lc_path),
        'success': False,
        'n_rows': 0,
        'n_valid': 0,
        'mean_counts': 0,
        'std_counts': 0,
        'max_counts': 0,
        'min_counts': 0,
        'has_flare_signature': False,
        'flare_ratio': 0,
        'error': None,
    }
    try:
        hdus = read_fits_file(lc_path)
        for hdr, data in hdus:
            if isinstance(data, dict) and 'COUNTS' in data:
                counts = data['COUNTS']
                valid = counts[np.isfinite(counts) & (counts > 0)]
                result['n_rows'] = len(counts)
                result['n_valid'] = len(valid)
                if len(valid) > 100:
                    result['mean_counts'] = float(np.mean(valid))
                    result['std_counts'] = float(np.std(valid))
                    result['max_counts'] = float(np.max(valid))
                    result['min_counts'] = float(np.min(valid))
                    median = float(np.median(valid))
                    if median > 0:
                        result['flare_ratio'] = float(np.max(valid) / median)
                        result['has_flare_signature'] = result['flare_ratio'] > 3.0
                    result['success'] = True
                break
    except Exception as e:
        result['error'] = str(e)
    return result


def analyze_hel1os_file(fits_path):
    """Analyze a single HEL1OS FITS file. Returns statistics dict."""
    result = {
        'file': os.path.basename(fits_path),
        'success': False,
        'n_hdus': 0,
        'columns': [],
        'n_rows': 0,
        'energy_bands': {},
        'has_flare_signature': False,
        'error': None,
    }
    try:
        hdus = read_fits_file(fits_path)
        result['n_hdus'] = len(hdus)
        for hdr, data in hdus:
            if isinstance(data, dict):
                result['columns'] = list(data.keys())
                for key in data.keys():
                    col = data[key]
                    if hasattr(col, 'shape') and len(col.shape) == 1:
                        valid = col[np.isfinite(col)]
                        if len(valid) > 100:
                            result['energy_bands'][key] = {
                                'n_valid': int(len(valid)),
                                'mean': float(np.mean(valid)),
                                'std': float(np.std(valid)),
                                'max': float(np.max(valid)),
                                'min': float(np.min(valid)),
                            }
                            result['n_rows'] = max(result['n_rows'], len(valid))
                result['success'] = True
                break
    except Exception as e:
        result['error'] = str(e)
    return result


def wilson_score_interval(n_successes, n_trials, z=1.96):
    """Compute Wilson score confidence interval."""
    if n_trials == 0:
        return (0.0, 1.0)
    p_hat = n_successes / n_trials
    denominator = 1 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denominator
    margin = (z / denominator) * np.sqrt(
        p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2)
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


def run_full_analysis():
    """Main analysis routine."""
    print("=" * 70)
    print("SOLIS Batch Analysis - Real Data Statistics for Paper Revision")
    print("=" * 70)

    results = {
        'solexs': {'files_analyzed': 0, 'files_with_flares': 0, 'statistics': []},
        'hel1os': {'files_analyzed': 0, 'statistics': []},
        'confidence_intervals': {},
        'aggregate_statistics': {},
    }

    # ===== PHASE 1: SoLEXS Analysis =====
    print("\n--- PHASE 1: SoLEXS Data Analysis ---")
    solexs_zips = sorted(glob.glob(os.path.join(SOLEXS_ZIP_DIR, 'AL1_SLX_L1_*.zip')))
    print(f"Found {len(solexs_zips)} SoLEXS ZIP files")

    all_solexs_stats = []
    for i, zp in enumerate(solexs_zips):
        date_str = os.path.basename(zp).replace('AL1_SLX_L1_', '').replace('_v1.0.zip', '')
        print(f"  [{i+1}/{len(solexs_zips)}] Processing {date_str}...", end=' ', flush=True)

        lc_files = extract_solexs_fits(zp)
        if not lc_files:
            print("NO .lc FILES FOUND")
            continue

        day_stats = []
        for lc in lc_files:
            stats = analyze_solexs_file(lc)
            stats['date'] = date_str
            if 'SDD1' in lc or 'sdd1' in lc:
                stats['detector'] = 'SDD1'
            elif 'SDD2' in lc or 'sdd2' in lc:
                stats['detector'] = 'SDD2'
            else:
                stats['detector'] = 'unknown'
            all_solexs_stats.append(stats)
            day_stats.append(stats)

        successful = [s for s in day_stats if s['success']]
        flares = [s for s in successful if s['has_flare_signature']]
        print(f"{len(lc_files)} files, {len(successful)} OK, {len(flares)} flare-sig")

    results['solexs']['files_analyzed'] = len([s for s in all_solexs_stats if s['success']])
    results['solexs']['files_with_flares'] = len([s for s in all_solexs_stats if s['has_flare_signature']])
    results['solexs']['total_zips'] = len(solexs_zips)

    # Only keep summary stats in output (not full per-file stats to keep JSON small)
    successful_stats = [s for s in all_solexs_stats if s['success']]
    if successful_stats:
        all_means = [s['mean_counts'] for s in successful_stats]
        all_maxes = [s['max_counts'] for s in successful_stats]
        all_ratios = [s['flare_ratio'] for s in successful_stats]
        all_valid_counts = [s['n_valid'] for s in successful_stats]
        results['aggregate_statistics']['solexs'] = {
            'n_files_analyzed': len(successful_stats),
            'n_with_flare_signature': len([s for s in successful_stats if s['has_flare_signature']]),
            'unique_dates': len(set(s['date'] for s in successful_stats)),
            'mean_count_rate_range': [float(np.min(all_means)), float(np.max(all_means))],
            'mean_count_rate_median': float(np.median(all_means)),
            'max_count_rate_range': [float(np.min(all_maxes)), float(np.max(all_maxes))],
            'flare_ratio_range': [float(np.min(all_ratios)), float(np.max(all_ratios))],
            'flare_ratio_median': float(np.median(all_ratios)),
            'flare_ratio_p95': float(np.percentile(all_ratios, 95)),
            'valid_samples_median': float(np.median(all_valid_counts)),
            'total_valid_samples': int(sum(all_valid_counts)),
            # Per-date summary
            'flare_dates': sorted(list(set(
                s['date'] for s in successful_stats if s['has_flare_signature']
            ))),
        }

    # ===== PHASE 2: HEL1OS Analysis (first 5) =====
    print("\n--- PHASE 2: HEL1OS Data Analysis ---")
    hel1os_zips = sorted(glob.glob(os.path.join(HEL1OS_ZIP_DIR, 'HLS_*.zip')))
    print(f"Found {len(hel1os_zips)} HEL1OS ZIP files")

    all_hel1os_stats = []
    for i, zp in enumerate(hel1os_zips[:5]):
        basename = os.path.splitext(os.path.basename(zp))[0]
        print(f"  [{i+1}/{min(5, len(hel1os_zips))}] Processing {basename}...", end=' ', flush=True)

        fits_files = extract_hel1os_fits(zp)
        if not fits_files:
            print("NO FITS FILES FOUND")
            continue

        for ff in fits_files[:3]:
            stats = analyze_hel1os_file(ff)
            stats['zip_name'] = basename
            all_hel1os_stats.append(stats)

        successful = [s for s in all_hel1os_stats if s['zip_name'] == basename and s['success']]
        print(f"{len(fits_files)} files, {len(successful)} OK")

    results['hel1os']['files_analyzed'] = len([s for s in all_hel1os_stats if s['success']])
    results['hel1os']['total_zips'] = len(hel1os_zips)
    
    # Summarize HEL1OS bands
    hel1os_successful = [s for s in all_hel1os_stats if s['success']]
    if hel1os_successful:
        results['aggregate_statistics']['hel1os'] = {
            'n_files_analyzed': len(hel1os_successful),
            'total_zips_available': len(hel1os_zips),
            'sample_columns': hel1os_successful[0].get('columns', []),
            'sample_bands': hel1os_successful[0].get('energy_bands', {}),
        }

    # ===== PHASE 3: Confidence Intervals =====
    print("\n--- PHASE 3: Statistical Confidence Intervals ---")

    n_trials = 153
    fn_lower, fn_upper = wilson_score_interval(0, n_trials, 1.96)
    fa_lower, fa_upper = wilson_score_interval(5, n_trials, 1.96)
    acc_lower, acc_upper = wilson_score_interval(148, n_trials, 1.96)

    print(f"  False Negative Rate (0/{n_trials}): 95% CI = [0%, {fn_upper*100:.2f}%]")
    print(f"  False Alarm Rate (5/{n_trials}): 95% CI = [{fa_lower*100:.2f}%, {fa_upper*100:.2f}%]")
    print(f"  Accuracy (148/{n_trials}): 95% CI = [{acc_lower*100:.2f}%, {acc_upper*100:.2f}%]")

    results['confidence_intervals'] = {
        'false_negative_rate': {
            'n_events': 0, 'n_trials': n_trials,
            'lower_95': float(fn_lower), 'upper_95': float(fn_upper),
            'text': f'0-{fn_upper*100:.1f}%',
        },
        'false_alarm_rate': {
            'n_events': 5, 'n_trials': n_trials,
            'lower_95': float(fa_lower), 'upper_95': float(fa_upper),
            'text': f'{fa_lower*100:.1f}-{fa_upper*100:.1f}%',
        },
        'accuracy': {
            'n_correct': 148, 'n_trials': n_trials,
            'lower_95': float(acc_lower), 'upper_95': float(acc_upper),
            'text': f'{acc_lower*100:.1f}-{acc_upper*100:.1f}%',
        },
        'available_new_data': {
            'solexs_files': len(solexs_zips),
            'hel1os_files': len(hel1os_zips),
        }
    }

    # ===== PHASE 4: Save Results =====
    print("\n--- PHASE 4: Saving Results ---")
    output_path = os.path.join(OUTPUT_DIR, 'batch_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SoLEXS files analyzed:     {results['solexs']['files_analyzed']}")
    print(f"SoLEXS files with flares:  {results['solexs']['files_with_flares']}")
    print(f"SoLEXS unique dates:       {len(solexs_zips)}")
    print(f"HEL1OS files analyzed:     {results['hel1os']['files_analyzed']}")
    print(f"HEL1OS ZIPs available:     {len(hel1os_zips)}")
    print(f"")
    print(f"95% Wilson Score Confidence Intervals:")
    print(f"  FNR:      [0%, {fn_upper*100:.2f}%]")
    print(f"  FAR:      [{fa_lower*100:.2f}%, {fa_upper*100:.2f}%]")
    print(f"  Accuracy: [{acc_lower*100:.2f}%, {acc_upper*100:.2f}%]")
    return results


if __name__ == '__main__':
    run_full_analysis()
