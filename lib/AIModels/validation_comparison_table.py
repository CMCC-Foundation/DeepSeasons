"""
Function to generate comparison LaTeX tables from validation CSV files.
"""

import pandas as pd


def comparison_scores_to_latex(testval_file, dynval_file, title="", region="", output_file=None, decimal_places=2):
    """
    Generate a LaTeX table comparing testval (DeepSeason) and dynval (GCM) validation scores.
    
    Creates a professional LaTeX table with dual columns for each metric (RPSS, BSS p1, p2, p3),
    comparing GCM (from dynval) and DeepSeason (from testval) results.
    
    Parameters
    ==========
    testval_file : str
        Path to the testval CSV file (DeepSeason results)
        Expected columns: Lead, RPSS(area), BSS(area) p1, BSS(area) p2, BSS(area) p3
        
    dynval_file : str
        Path to the dynval CSV file (GCM results)
        Expected columns: Lead, RPSS(area), BSS(area) p1, BSS(area) p2, BSS(area) p3
        
    title : str
        Title for the table caption (e.g., "Global SST")
        
    region : str
        Region description for the caption (e.g., "Area-aggregated")
        
    output_file : str or None
        Path to write the LaTeX table. If None, generates a default filename
        
    decimal_places : int
        Number of decimal places for rounding (default: 2)
        
    Returns
    =======
    latex_table : str
        LaTeX formatted table
        
    Examples
    ========
    >>> table = comparison_scores_to_latex(
    ...     'testval_Y621s.csv',
    ...     'dynval_Y621s.csv',
    ...     title='Global SST',
    ...     region='Area-aggregated',
    ...     output_file='comparison_table.tex'
    ... )
    """
    
    import os
    
    # Read both CSV files
    test_df = pd.read_csv(testval_file)
    dyn_df = pd.read_csv(dynval_file)
    
    # Merge on Lead column
    merged_df = pd.merge(test_df, dyn_df, on='Lead', suffixes=('_test', '_dyn'), how='outer')
    merged_df = merged_df.sort_values('Lead')
    
    # Extract columns
    lead_col = merged_df['Lead']
    
    # RPSS columns
    rpss_dyn = merged_df['RPSS(area)_dyn']
    rpss_test = merged_df['RPSS(area)_test']
    
    # BSS columns
    bss_p1_dyn = merged_df['BSS(area) p1_dyn']
    bss_p1_test = merged_df['BSS(area) p1_test']
    
    bss_p2_dyn = merged_df['BSS(area) p2_dyn']
    bss_p2_test = merged_df['BSS(area) p2_test']
    
    bss_p3_dyn = merged_df['BSS(area) p3_dyn']
    bss_p3_test = merged_df['BSS(area) p3_test']
    
    # Build LaTeX table
    latex_lines = []
    
    latex_lines.append('\\begin{table}[ht]')
    latex_lines.append('\\centering')
    
    # Caption
    caption = f'\\textbf{{{title}}}. {region} RPSS and Brier Skill Scores (BSS) for tercile events '
    caption += '($p_1$, $p_2$, $p_3$) as a function of lead time. Each metric is shown for the '
    caption += 'operational GCM and DeepSeason.'
    latex_lines.append(f'\\caption{{{caption}}}')
    latex_lines.append(f'\\label{{tab:{title.lower().replace(" ", "_")}}}')
    
    # Table opening with column specification
    latex_lines.append('\\begin{tabular}{r*{8}{r}}')
    latex_lines.append('\\toprule')
    
    # Header row 1: Metric names with multicolumn
    header1 = ' & \\multicolumn{2}{c}{RPSS} & \\multicolumn{2}{c}{BSS($p_1$)} & \\multicolumn{2}{c}{BSS($p_2$)} & \\multicolumn{2}{c}{BSS($p_3$)} \\\\'
    latex_lines.append(header1)
    
    # Partial rules after metric names
    latex_lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}')
    
    # Header row 2: Column names
    header2 = 'Lead & GCM & DeepSeason & GCM & DeepSeason & GCM & DeepSeason & GCM & DeepSeason \\\\'
    latex_lines.append(header2)
    latex_lines.append('\\midrule')
    
    # Data rows
    for idx, row in merged_df.iterrows():
        lead = int(row['Lead'])
        
        # Build row string
        row_str = str(lead)
        
        # RPSS columns
        rpss_dyn_val = row['RPSS(area)_dyn']
        rpss_test_val = row['RPSS(area)_test']
        row_str += f' & {rpss_dyn_val:.{decimal_places}f}' if pd.notna(rpss_dyn_val) else ' & '
        row_str += f' & {rpss_test_val:.{decimal_places}f}' if pd.notna(rpss_test_val) else ' & '
        
        # BSS p1 columns
        bss_p1_dyn_val = row['BSS(area) p1_dyn']
        bss_p1_test_val = row['BSS(area) p1_test']
        row_str += f' & {bss_p1_dyn_val:.{decimal_places}f}' if pd.notna(bss_p1_dyn_val) else ' & '
        row_str += f' & {bss_p1_test_val:.{decimal_places}f}' if pd.notna(bss_p1_test_val) else ' & '
        
        # BSS p2 columns
        bss_p2_dyn_val = row['BSS(area) p2_dyn']
        bss_p2_test_val = row['BSS(area) p2_test']
        row_str += f' & {bss_p2_dyn_val:.{decimal_places}f}' if pd.notna(bss_p2_dyn_val) else ' & '
        row_str += f' & {bss_p2_test_val:.{decimal_places}f}' if pd.notna(bss_p2_test_val) else ' & '
        
        # BSS p3 columns
        bss_p3_dyn_val = row['BSS(area) p3_dyn']
        bss_p3_test_val = row['BSS(area) p3_test']
        row_str += f' & {bss_p3_dyn_val:.{decimal_places}f}' if pd.notna(bss_p3_dyn_val) else ' & '
        row_str += f' & {bss_p3_test_val:.{decimal_places}f}' if pd.notna(bss_p3_test_val) else ' & '
        
        row_str += ' \\\\'
        latex_lines.append(row_str)
    
    # Table closing
    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')
    
    latex_table = '\n'.join(latex_lines)
    
    # Generate default output filename if not specified
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(testval_file))[0]
        output_file = f"{base_name}_comparison.tex"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {output_file}")
    
    return latex_table
