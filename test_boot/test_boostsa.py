# coding=latin-1
from boostsa import Bootstrap

def test_return_p_values_hard_labels():
    boot = Bootstrap()
    df_tot, df_tgt = boot.test(targs='test_boot/h0.0/targs.txt', h0_preds='test_boot/h0.0/preds.txt', h1_preds='test_boot/h1.0/preds.txt', n_loops=100)
    assert 'p_f1' in df_tot.columns
    assert 'p_acc' in df_tot.columns
    assert 'p_prec' in df_tot.columns
    assert 'p_rec' in df_tot.columns
    # p values should be NaN or float values between 0.0 and 1.0
    assert all(df_tot['p_f1'].isna() | ((df_tot['p_f1'] >= 0.0) & (df_tot['p_f1'] <= 1.0)))
    assert all(df_tot['p_acc'].isna() | ((df_tot['p_acc'] >= 0.0) & (df_tot['p_acc'] <= 1.0)))
    assert all(df_tot['p_prec'].isna() | ((df_tot['p_prec'] >= 0.0) & (df_tot['p_prec'] <= 1.0)))
    assert all(df_tot['p_rec'].isna() | ((df_tot['p_rec'] >= 0.0) & (df_tot['p_rec'] <= 1.0)))

def test_return_p_values_soft_labels():
    boot = Bootstrap()
    df_tot, _ = boot.test(targs='test_boot/h3.0/targs.txt', h0_preds='test_boot/h3.0/preds.txt', h1_preds='test_boot/h3.0/preds.txt', n_loops=100)
    assert 'p_jsd' in df_tot.columns
    assert 'p_ce' in df_tot.columns
    assert 'p_sim' in df_tot.columns
    assert 'p_cor' in df_tot.columns
    # p values should be NaN or float values between 0.0 and 1.0
    assert all(df_tot['p_jsd'].isna() | ((df_tot['p_jsd'] >= 0.0) & (df_tot['p_jsd'] <= 1.0)))
    assert all(df_tot['p_ce'].isna() | ((df_tot['p_ce'] >= 0.0) & (df_tot['p_ce'] <= 1.0)))
    assert all(df_tot['p_sim'].isna() | ((df_tot['p_sim'] >= 0.0) & (df_tot['p_sim'] <= 1.0)))
    assert all(df_tot['p_cor'].isna() | ((df_tot['p_cor'] >= 0.0) & (df_tot['p_cor'] <= 1.0)))

def test_all():
    boot = Bootstrap()
    boot.test(targs='test_boot/h0.0/targs.txt', h0_preds='test_boot/h0.0/preds.txt', h1_preds='test_boot/h1.0/preds.txt')

    boot = Bootstrap()

    boot.feed(h0='h0',          exp_idx='h0.0', preds='test_boot/h0.0/preds.txt', targs='test_boot/h0.0/targs.txt', idxs='test_boot/h0.0/idxs.txt')
    boot.feed(h0='h0',          exp_idx='h0.1', preds='test_boot/h0.1/preds.txt', targs='test_boot/h0.1/targs.txt', idxs='test_boot/h0.1/idxs.txt')
    boot.feed(h0='h0', h1='h1', exp_idx='h1.0', preds='test_boot/h1.0/preds.txt', targs='test_boot/h1.0/targs.txt', idxs='test_boot/h1.0/idxs.txt')
    boot.feed(h0='h0', h1='h1', exp_idx='h1.1', preds='test_boot/h1.1/preds.txt', targs='test_boot/h1.1/targs.txt', idxs='test_boot/h1.1/idxs.txt')
    boot.run(n_loops=100, sample_size=.2, verbose=True)

    next_boot = Bootstrap()
    next_boot.loadjson('outcomes.json')

    next_boot.feed(h0='h0', h1='h2', exp_idx='h2.0', preds='test_boot/h2.0/preds.txt', targs='test_boot/h2.0/targs.txt', idxs='test_boot/h2.0/idxs.txt')
    next_boot.feed(h0='h0', h1='h2', exp_idx='h2.1', preds='test_boot/h2.1/preds.txt', targs='test_boot/h2.1/targs.txt', idxs='test_boot/h2.1/idxs.txt')
    next_boot.run(n_loops=100, sample_size=.2, verbose=True)