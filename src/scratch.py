from module1_var import fit_match_level
from module2_shootout import run_test
from module3_format_ab import run_experiment
ml=fit_match_level()
sh=run_test()
ex=run_experiment()
r=ex['results']
print(f"M1 IRR {ml['irr']:.2f}x p={ml['p_value']:.4f}")
print(f"M2 {sh['win_rate']*100:.1f}% n={sh['n_shootouts']} p={sh['p_value']:.3f}")
print(f"M3 champ {r['champion_elo']['mean_A_32team']:.0f} vs {r['champion_elo']['mean_B_48team']:.0f} | upsets {r['knockout_upsets']['mean_A_32team']:.1f} vs {r['knockout_upsets']['mean_B_48team']:.1f}")