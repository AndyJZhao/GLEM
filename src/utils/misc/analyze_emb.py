def _analyze_data(self):
    # ! Save LM embedding for plot
    self.log(f'\n\n <<<<<<<<<< CT-ITER - {self.iteration} ANALYSIS >>>>>>>>>>')

    EMB_DIM = 7
    SAMPLE_POINTS = 200
    if self.cf.analyze == 'PlotEmb':
        import matplotlib.pyplot as plt
        import seaborn as sns

        _load = lambda f: th.load(f)[:, :EMB_DIM].cpu().numpy()
        emb_lookup = {
            'LM-Enc': lambda i: _load(self.cf.lm_output_emb(i - 1)),
            # 'LM-Dec': lambda i, dim: self.cf.lm_output_emb(i)[dim].cpu().numpy(),
            'GNN': lambda i: _load(self.cf.gnn_output_emb(i)),
        }

        def get_emb_df(ct_iter):
            emb_dict = {module_name: lookup_func(ct_iter)
                        for module_name, lookup_func in emb_lookup.items()}
            df = pd.concat([pd.DataFrame(emb) for emb in emb_dict.values()], keys=emb_dict.keys())
            df['Module'] = df.index.get_level_values(0)
            df['CT-Epoch'] = f'{ct_iter}'
            sample_rate = SAMPLE_POINTS / len(df)
            return df.sample(frac=sample_rate) if sample_rate < 1 else df

        # ! Prepare Data
        # all_dim_data = pd.concat([get_emb_df(ct_iter) for ct_iter in range(1, self.cf.ct_n_iters + 2)])

        all_dim_data = pd.concat([get_emb_df(ct_iter) for ct_iter in range(1, 3)])
        plt_df = pd.concat([all_dim_data[[dim, 'Module', 'CT-Iter']].rename({dim: 'Emb Distribution'}, axis=1) for dim in range(EMB_DIM)], keys=range(EMB_DIM))
        plt_df['Emb Dim'] = plt_df.index.get_level_values(0)
        plt_file = self.cf.res_file.replace(RES_PATH, 'plots/').replace('.json', f'EmbAnalysis.jpg')
        y_min, y_max = (_ := plt_df['Emb Distribution']).min(), _.max()
        # ! Plot
        fig, axes = plt.subplots(2, 1, figsize=(18, 18))
        # fig.suptitle('Embedding distribution')
        for _, module in enumerate(emb_lookup.keys()):
            ax = sns.violinplot(
                ax=axes[_], data=plt_df.query(f"Module=='{module}'", engine='python'),
                x='Emb Dim', y='Emb Distribution', hue='CT-Iter', palette="Set2", split=True,
                scale="count", inner=None, scale_hue=False)
            ax.set(ylim=(y_min, y_max))
            ax.set_title(module)
        plt.show()
        fig.savefig(plt_file)
        self.log(f'Figure saved to {plt_file}')
        self.logger.log_fig('Emb Visualization', plt_file)
