### 9.4 Run label transfer (optional `"manual"`` mode)

- Manual mode allows users to inspect clustering, anchor detection, and label assignment before committing each hierarchy split.

- In manual mode, `run_label_transfer_stage()` returns a session for each query section rather than a finalized transfer result. Final CSV, H5AD, and postprocessing outputs are created only after converting the session with `session.to_result()` and saving the outputs.

#### 9.4.1 Initialize a manual session

```python
manual_stage_result = run_label_transfer_stage(
    jobs=job_setup.jobs,
    config=LabelTransferStageConfig(
        scenario=job_setup.scenario,
        output_dir=analysis_root / "06_label_transfer",
        mode="manual",
        parameters={
            "label_key": label_key,
            "cluster_key": "query_cluster",
            "final_label_key": "hicat_label",
            "unassigned_label": "novel_cluster",
            "min_node_prop": 0.05,
            "min_node_spots": 2,
            "copy": False,
            "boundary_refinement_config": hipt_boundary_refinement_config,
            "gene_subtyping_config": gene_subtyping_config,
            "anchor_config": anchor_config,
            "assignment_config": assignment_config,
            "print_results": True,
        },
        save_intermediate_figures=True,
        intermediate_figure_parameters=intermediate_figure_paras,
        postprocess=True,
        postprocess_parameters=postprocess_paras,
    ),
)

query_section = query_sections[0]
session = manual_stage_result.get_result(query_section)

print("Start node:", session.start_node)
print("Pending nodes:", session.pending_internal_nodes())
```

#### 9.4.2 Preview and inspect one hierarchy round

```python
parent_node = session.pending_internal_nodes()[0]

round_preview = session.run_round(
    parent_node=parent_node,
    commit=False,
)

print(round_preview.summary())
print(round_preview.clustering_result.labels.value_counts())
print(round_preview.anchor_result.anchor_df.head())
print(round_preview.assignment_result.labels.value_counts())
print(round_preview.assignment_result.cross_table)
print(round_preview.assignment_result.adjusted_cross_table)
```

#### 9.4.3 Commit an accepted round

```python
session.commit_round(round_preview)

print("Pending nodes after commit:")
print(session.pending_internal_nodes())
```

Continue through the hierarchy:

```python
while session.pending_internal_nodes():
    parent_node = session.pending_internal_nodes()[0]

    round_preview = session.run_round(
        parent_node=parent_node,
        commit=False,
    )

    print(round_preview.summary())
    print(round_preview.clustering_result.labels.value_counts())
    print(round_preview.assignment_result.labels.value_counts())

    # Commit the round after inspecting its outputs.
    session.commit_round(round_preview)
```

#### 9.4.4 Finalize and save manual-mode results

```python
manual_result = session.to_result(mode="manual")

print(manual_result.final_labels.value_counts(dropna=False))
print("Complete:", manual_result.is_complete)
print("Pending nodes:", manual_result.pending_nodes)
```

```python
from hicat_spatial.label_transfer import save_label_transfer_outputs

manual_gene_postprocessed = save_label_transfer_outputs(
    transfer_result=manual_result,
    transfer_scenario=job_setup.scenario,
    output_dir=analysis_root / "06_label_transfer",
    qry_section=query_section,
    **postprocess_paras,
)
```

#### 9.4.5 Tune an individual round when needed

```python
parent_node = session.pending_internal_nodes()[0]

round_preview = session.run_round(
    parent_node=parent_node,
    clustering_overrides={
        "resolution": 0.05,
        "n_neighbors": 30,
    },
    assignment_overrides={
        "min_anchor_pct": 3,
        "prop_diff_cutoff": 10,
    },
    commit=False,
)

print(round_preview.summary())
```

Commit the tuned round when its results are satisfactory:

```python
session.commit_round(round_preview)
```

---