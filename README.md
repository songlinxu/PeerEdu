# [CHI 2025] PeerEdu: Bootstrapping Online Learning Behaviors via Asynchronous Area of Interest Sharing from Peer Gaze

# The instructions below are old figures. For updated ones in our final manuscript, we will update soon.

Please note that student_demo.csv directly records student demographics from survey. As a result, there may be duplicated student_id if students submit the demographic data for more than one time. However, in other files, there may be new student_id's behavioral data whose student_id is not available in student_demo.csv because some students particiated the study without submitting their demographic data. As such, we suggest you to find overlapped student_ids between student_demo.csv and other data files for simulation, and then make all student_ids unique using set().

# To draw figures in the paper 

- Use figure_plot.py
- Run f2() in figure_plot.py to draw figure 2 in the paper
- Run f3() in figure_plot.py to draw figure 3 in the paper
- Run f4() in figure_plot.py to draw figure 4 in the paper


# To replicate results reported in the paper

- Use result_analysis.py
- Run s1_gaze_manipulate() in result_analysis.py to replicate results in the gaze manipulation subsection in Results section in the paper.
- Run s2_learn_experience() in result_analysis.py to replicate results in the learning experience subsection in Results section in the paper.
- Run s3_learn_outcome() in result_analysis.py to replicate results in the learning outcome subsection in Results section in the paper.
- Run s4_decode_learn() in result_analysis.py to replicate results in the decoding learning process subsection in Results section in the paper.

# Cite us

Please cite our paper if you find our datasets/codes useful.
```bibtex
@article{xu2023peer,
  title={Peer attention enhances student learning},
  author={Xu, Songlin and Hu, Dongyin and Wang, Ru and Zhang, Xinyu},
  journal={arXiv preprint arXiv:2312.02358},
  year={2023}
}
```
