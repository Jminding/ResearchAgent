# Complete Source List: Quantum Error Correction Literature Survey

**Survey Date**: December 28, 2025
**Total Sources**: 40+ papers, reports, and technical articles
**Format**: Complete citations with URLs for direct access

---

## Primary Hardware Demonstrations

### Google Quantum AI - Willow Chip (2024)

1. **[Google2024_Willow]** "Quantum error correction below the surface code threshold"
   - Venue: Nature (In press; preprint available August 2024)
   - Authors: Google Quantum AI team
   - URL: https://arxiv.org/abs/2408.13687
   - Key metrics: Λ = 2.14 ± 0.02×, logical error 0.143% ± 0.003%, 2.4× lifetime advantage
   - Significance: First below-threshold exponential error suppression on hardware

2. **[Google2024_Willow_Blog]** "Making quantum error correction work"
   - Venue: Google Research Blog
   - URL: https://research.google/blog/making-quantum-error-correction-work/
   - Description: Technical explanation of Willow results

3. **[GoogleWillow_NextPlatform]** "Google Claims Quantum Error Correction Milestone With Willow Chip"
   - Venue: Next Platform
   - URL: https://www.nextplatform.com/2024/12/09/google-claims-quantum-error-correction-milestone-with-willow-chip/

4. **[GoogleWillow_HPCWire]** "Google debuts next-gen quantum computing chip with breakthrough error correction"
   - Venue: HPCWire
   - URL: https://www.hpcwire.com/2024/12/09/google-debuts-next-gen-quantum-computing-chip-error-correction-breakthrough-and-roadmap-details/

### Harvard/MIT Neutral Atom Processor (2023)

5. **[Harvard_MIT_2023]** "Logical quantum processor based on reconfigurable atom arrays"
   - Authors: Bluvstein et al. (Harvard, MIT, QuEra)
   - Venue: Nature, December 2023
   - URL: https://www.nature.com/articles/s41586-023-06927-3
   - Key metrics: 48 logical qubits, 280 physical qubits, algorithms outperform physical
   - Significance: First large-scale error-corrected algorithm execution

6. **[Harvard_Gazette]** "Harvard researchers create first logical quantum processor"
   - Venue: Harvard Gazette
   - URL: https://news.harvard.edu/gazette/story/2023/12/researchers-create-first-logical-quantum-processor/

7. **[QuEra_Press_Release]** "Error-Corrected Quantum Algorithms on 48 Logical Qubits"
   - Venue: QuEra press release
   - URL: https://www.quera.com/press-releases/harvard-quera-mit-and-the-nist-university-of-maryland-usher-in-new-era-of-quantum-computing-by-performing-complex-error-corrected-quantum-algorithms-on-48-logical-qubits0

### Google DeepMind - AlphaQubit (2024)

8. **[AlphaQubit_2024]** "Learning high-accuracy error decoding for quantum processors"
   - Authors: Google DeepMind
   - Venue: Nature (Published November 2024)
   - URL: https://www.nature.com/articles/s41586-024-08148-8
   - Key metrics: 30% improvement vs SCAM, 6% vs TN methods, d≤11 tested
   - Significance: Transformer-based neural decoder validates on real hardware

9. **[AlphaQubit_Blog]** "AlphaQubit: Google's research on quantum error correction"
   - Venue: Google Research Blog
   - URL: https://blog.google/technology/google-deepmind/alphaqubit-quantum-error-correction/

10. **[AlphaQubit_eWEEK]** "Google's AlphaQubit Breakthrough: 6% Better Quantum Error Detection"
    - Venue: eWEEK
    - URL: https://www.eweek.com/news/alphaqubit-boosts-quantum-error-detection/

11. **[AlphaQubit_Medium]** "How Google AI Used Machine Learning for Quantum Error Correction"
    - Author: Devansh
    - Venue: Medium (ODSC)
    - URL: https://machine-learning-made-simple.medium.com/how-google-ai-used-machine-learning-for-quantum-error-correction-b7c927e0e17b

### IBM Quantum - QLDPC and Gross Codes (2024)

12. **[IBM_QLDPC_2024]** "Building the future of quantum error correction"
    - Venue: IBM Quantum Computing Blog
    - URL: https://www.ibm.com/quantum/blog/future-quantum-error-correction

13. **[IBM_Gross_Nature]** "Landmark IBM error correction paper on Nature cover"
    - Venue: IBM Quantum Computing Blog (QLDPC codes published in Nature)
    - URL: https://www.ibm.com/quantum/blog/nature-qldpc-error-correction

14. **[IBM_Relay_BP]** "Introducing Relay-BP"
    - Venue: IBM Quantum Computing Blog
    - URL: https://www.ibm.com/quantum/blog/relay-bp-error-correction-decoder
    - Description: Orders of magnitude improvement for qLDPC decoders

15. **[IBM_QLDPC_Computing]** "Computing with error-corrected quantum computers"
    - Venue: IBM Quantum Computing Blog
    - URL: https://www.ibm.com/quantum/blog/qldpc-codes

16. **[IBM_Fault_Tolerant_2024]** "IBM lays out clear path to fault-tolerant quantum computing"
    - Venue: IBM Quantum Computing Blog
    - URL: https://www.ibm.com/quantum/blog/large-scale-ftqc

### Quantinuum/Microsoft - Trapped-Ion QEC (2024)

17. **[Quantinuum_Microsoft_2024]** "Logical error suppression on Quantinuum H2"
    - Partnership: Quantinuum + Microsoft
    - Date: April 2024
    - Key metric: 800× logical-to-physical error ratio
    - Reference: https://ionq.com/blog/our-novel-efficient-approach-to-quantum-error-correction

18. **[IonQ_Blog]** "Our Novel, Efficient Approach to Quantum Error Correction"
    - Venue: IonQ Blog
    - URL: https://ionq.com/blog/our-novel-efficient-approach-to-quantum-error-correction

### Other Hardware Platforms

19. **[Oxford_Ionics_2025]** "99.99% fidelity two-qubit gates"
    - Venue: Riverlane blog (2024's QEC Highlights)
    - URL: https://www.riverlane.com/blog/2024-s-qec-highlights-aka-the-12-days-of-qechristmas

20. **[China_Microwave_QEC_2025]** "China Demonstrates Quantum Error Correction Using Microwaves"
    - Venue: The Quantum Insider (December 2025)
    - URL: https://thequantuminsider.com/2025/12/26/china-demonstrates-quantum-error-correction-using-microwaves-narrowing-gap-with-google/

---

## Machine Learning and Neural Decoder Papers

### Comprehensive Benchmarking Studies

21. **[Benchmarking_ML_2024]** "Benchmarking Machine Learning Models for Quantum Error Correction"
    - Venue: ICML/NeurIPS / OpenReview / arXiv 2311.11167v3
    - Date: 2024
    - URL: https://arxiv.org/abs/2311.11167
    - Key findings: 7 architectures compared; U-Net 50% vs CNN; performance doesn't degrade with distance

22. **[ML_QEM_2024]** "Machine Learning for Practical Quantum Error Mitigation"
    - Authors: IBM Research + collaborators
    - Venue: arXiv 2309.17368
    - Date: 2024
    - URL: https://arxiv.org/abs/2309.17368
    - Key findings: Random forests best; competitive with ZNE at lower cost

23. **[Scalable_Neural_Decoders]** "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction"
    - Venue: arXiv 2510.22724
    - Date: 2025
    - URL: https://arxiv.org/html/2510.22724v1

### Real-Time Decoding and Hardware Implementation

24. **[RealTime_QEC_2024]** "Demonstrating real-time and low-latency quantum error correction with superconducting qubits"
    - Venue: arXiv 2410.05202
    - Date: October 2024
    - URL: https://arxiv.org/abs/2410.05202

25. **[Riverlane_RealTime_2025]** "Riverlane unveils microsecond quantum error decoder"
    - Venue: IT Brief
    - Date: December 2025
    - URL: https://itbrief.asia/story/riverlane-unveils-microsecond-quantum-error-decoder/

26. **[Riverlane_NatComm]** "Local Clustering Decoder" (Nature Communications)
    - Publisher: Nature Communications
    - Date: 2024
    - Description: Hardware-based real-time QEC decoder for surface codes
    - Deployment: Infleqtion, Oxford Quantum Circuits, Rigetti, ORNL

27. **[QUEKUF_FPGA]** "QUEKUF: An FPGA Union Find Decoder for Quantum Error Correction on the Toric Code"
    - Venue: ACM Transactions on Reconfigurable Technology and Systems
    - Date: 2024
    - URL: https://dl.acm.org/doi/10.1145/3733239
    - Key metrics: 7.30× speedup vs C++, 81.51× energy efficiency

---

## Reinforcement Learning for QEC

### Code Optimization and Discovery

28. **[Nautrup_RL_2019]** "Optimizing Quantum Error Correction Codes with Reinforcement Learning"
    - Authors: Nautrup et al.
    - Venue: Quantum Journal, 2019
    - URL: https://quantum-journal.org/papers/q-2019-12-16-215/
    - Key findings: RL discovers near-optimal surface codes; transfers across noise models

29. **[Tomasini_RL_ToricCode_2019]** "Quantum error correction for the toric code using deep reinforcement learning"
    - Authors: Tomasini et al.
    - Venue: Quantum Journal, 2019
    - URL: https://quantum-journal.org/papers/q-2019-09-02-183/
    - Key findings: Threshold ~11% (near-optimal); outperforms MWPM at high error rates

30. **[NPJ_RL_CodeDiscovery_2024]** "Simultaneous discovery of quantum error correction codes and encoders with a noise-aware reinforcement learning agent"
    - Venue: npj Quantum Information, 2024
    - URL: https://www.nature.com/articles/s41534-024-00920-y
    - Key findings: Discovers codes and encoding circuits; works up to d=5, 25 qubits

31. **[OpenReview_RL_Scaling]** "Scaling Automated Quantum Error Correction Discovery with Reinforcement Learning"
    - Venue: OpenReview
    - URL: https://openreview.net/forum?id=PP40WPYr3F

### RL for Real-Time Control and Adaptation

32. **[RL_RealTimeControl_2025]** "Reinforcement Learning Control of Quantum Error Correction"
    - Venue: arXiv 2511.08493
    - Date: November 2025
    - URL: https://arxiv.org/abs/2511.08493
    - Key findings: 3.5× stability improvement against parameter drift

33. **[RL_HeavyHexagon]** "Quantum error correction for heavy hexagonal code using deep reinforcement learning with policy reuse"
    - Venue: Quantum Information Processing
    - Date: 2024
    - URL: https://link.springer.com/article/10.1007/s11128-024-04377-y

34. **[RL_Autonomous]** "Discovering autonomous quantum error correction via deep reinforcement learning"
    - Venue: Phys. Rev. A
    - URL: https://link.aps.org/doi/10.1103/rgy3-z928

---

## Industry Reports and Survey Papers

### Riverlane QEC Reports

35. **[Riverlane_Report_2024]** "The Quantum Error Correction Report 2024"
    - Publisher: Riverlane Research
    - Date: 2024
    - URL: https://www.riverlane.com/quantum-error-correction-report-2024
    - Coverage: Industry trends, hardware milestones, timeline predictions

36. **[Riverlane_2025_Trends]** "Quantum Error Correction: Our 2025 trends and 2026 predictions"
    - Venue: Riverlane Blog
    - Date: 2025
    - URL: https://www.riverlane.com/blog/quantum-error-correction-our-2025-trends-and-2026-predictions

37. **[Riverlane_2024_Highlights]** "2024's Quantum Error Correction Highlights (aka the 12 Days of QEChristmas)"
    - Venue: Riverlane Blog
    - Date: December 2024
    - URL: https://www.riverlane.com/blog/2024-s-qec-highlights-aka-the-12-days-of-qechristmas

38. **[Riverlane_Challenge]** "Riverlane report reveals scale of the Quantum Error Correction challenge"
    - Venue: Riverlane Press Release
    - URL: https://www.riverlane.com/press-release/riverlane-report-reveals-scale-of-the-quantum-error-correction-challenge

### Other Industry and Academic Reviews

39. **[QuantumZeitgeist_Report]** "Quantum Error Correction Report 2025"
    - Venue: Quantum Zeitgeist
    - URL: https://quantumzeitgeist.com/quantum-error-correction/

40. **[QuantumInsider_2025_Trends]** "Report: Error Correction Becomes 'Universal Priority' But Talent Shortage Looms"
    - Venue: The Quantum Insider
    - Date: November 2025
    - URL: https://thequantuminsider.com/2025/11/19/report-error-correction-becomes-universal-priority-but-talent-shortage-looms/

41. **[QuantumInsider_Willow]** "Google Quantum AI: New Quantum Chip Outperforms Classical Computers and Breaks Error Correction Threshold"
    - Venue: The Quantum Insider
    - Date: December 2024
    - URL: https://thequantuminsider.com/2024/12/09/google-quantum-ai-new-quantum-chip-outperforms-classical-computers-and-breaks-error-correction-threshold/

42. **[OReilly_Update_2024]** "Quantum Error Correction Update 2024"
    - Venue: O'Reilly
    - URL: https://www.oreilly.com/radar/quantum-error-correction-update-2024/

43. **[IEEE_Spectrum]** "Quantum Error Correction: Time to Make It Work"
    - Venue: IEEE Spectrum
    - URL: https://spectrum.ieee.org/quantum-error-correction

44. **[Physics_World_2024]** "Two advances in quantum error correction share the Physics World 2024 Breakthrough of the Year"
    - Venue: Physics World
    - Date: December 2024
    - URL: https://physicsworld.com/a/two-advances-in-quantum-error-correction-share-the-physics-world-2024-breakthrough-of-the-year/

45. **[Quanta_Magazine]** "Quantum Computers Cross Critical Error Threshold"
    - Venue: Quanta Magazine
    - Date: December 2024
    - URL: https://www.quantamagazine.org/quantum-computers-cross-critical-error-threshold-20241209/

---

## Specialized Topic Papers

### Trapped-Ion and Scaling

46. **[TrappedIon_Scaling_2025]** "Quantum error correction for long chains of trapped ions"
    - Venue: Quantum Journal
    - Date: 2025
    - URL: https://quantum-journal.org/papers/q-2025-11-27-1920/
    - Key findings: BB5 codes 10-20× improvement; lattice surgery scaling

47. **[IonTrap_Architecture]** "Ion-Trap Chip Architecture Optimized for Implementation of Quantum Error-Correcting Code"
    - Venue: arXiv 2501.15200
    - Date: January 2025
    - URL: https://arxiv.org/html/2501.15200

48. **[Crosstalk_Analysis]** "Performance Analysis for Crosstalk Errors between Parallel Entangling Gates in Trapped Ion Quantum Error Correction"
    - Venue: arXiv 2501.09554
    - Date: January 2025
    - URL: https://arxiv.org/html/2501.09554

### Decoder Scheduling and Real-Time Challenges

49. **[CODA_Scheduling]** "Constraint-Optimal Driven Allocation for Scalable QEC Decoder Scheduling"
    - Venue: arXiv 2512.02539
    - Date: December 2025
    - URL: https://arxiv.org/html/2512.02539
    - Key findings: 74% reduction in undecoded sequence length; linear scaling

50. **[Qrisp_Analysis]** "Analysis of Surface Code Algorithms on Quantum Hardware Using the Qrisp Framework"
    - Venue: Electronics, 14(23), 4707
    - Date: 2024
    - URL: https://www.mdpi.com/2079-9282/14/23/4707

### Adversarial and Robustness Issues

51. **[Adversarial_QEC]** "Fooling the Decoder: An Adversarial Attack on Quantum Error Correction"
    - Venue: arXiv 2504.19651
    - Date: April 2025
    - URL: https://arxiv.org/html/2504.19651
    - Key finding: Neural decoders vulnerable to adversarial syndrome patterns

### Limitations and Theoretical Analysis

52. **[Limitations_ErrorMitigation]** "Exponentially tighter bounds on limitations of quantum error mitigation"
    - Venue: Nature Physics
    - Date: 2024
    - URL: https://www.nature.com/articles/s41567-024-02536-7

53. **[Open_System_Dynamics]** "Quantum error correction under numerically exact open-quantum-system dynamics"
    - Venue: Phys. Rev. Research, 5, 043161
    - Date: 2024
    - URL: https://link.aps.org/doi/10.1103/PhysRevResearch.5.043161

---

## Open-Source Tools and Benchmarking Resources

### Software Frameworks

54. **[MQT_QECC]** MQT QECC - Tools for Quantum Error Correcting Codes
    - Repository: https://github.com/munich-quantum-toolkit/qecc
    - Purpose: Code simulation and MaxSAT-based decoding

55. **[Infleqtion_qLDPC]** "Infleqtion Unveils Open-Source Library for Quantum Error Correction Research"
    - Venue: Infleqtion press release
    - URL: https://infleqtion.com/infleqtion-unveils-open-source-library-for-quantum-error-correction-research/

56. **[NVIDIA_CUDA_QX]** "Streamlining Quantum Error Correction with CUDA-QX 0.4"
    - Venue: NVIDIA Technical Blog
    - URL: https://developer.nvidia.com/blog/streamlining-quantum-error-correction-and-application-development-with-cuda-qx-0-4/

57. **[Stim_pyMatching_Tutorial]** "How to benchmark your first quantum error correction experiment using stim and pyMatching"
    - Author: Quantum for the Confused (Medium)
    - URL: https://quantum-for-the-confused.medium.com/how-to-benchmark-your-first-quantum-error-correction-experiment-using-stim-and-pymatching-786301b2452d

### Databases and References

58. **[ErrorCorrectionZoo]** Error Correction Zoo
    - URL: https://errorcorrectionzoo.org
    - Purpose: Comprehensive database of QEC codes and their properties
    - Coverage: 1000+ codes; threshold benchmarks

---

## Supporting Articles and News

59. **[Qblox_News]** "The quantum leap that needs error correction"
    - Venue: Qblox
    - URL: https://qblox.com/newsroom/the-quantum-leap-that-needs-error-correction

60. **[MIT_Tech_Review]** "A new ion-based quantum computer makes error correction simpler"
    - Venue: MIT Technology Review
    - Date: November 2025
    - URL: https://www.technologyreview.com/2025/11/05/1127659/a-new-ion-based-quantum-computer-makes-error-correction-simpler/

61. **[Quantum_Machines_Blog]** "Google's Quantum Error Correction Breakthrough"
    - Venue: Quantum Machines Blog
    - URL: https://www.quantum-machines.co/blog/understanding-googles-quantum-error-correction-breakthrough/

62. **[SiliconANGLE]** "Google DeepMind's AlphaQubit tackles quantum error detection with unprecedented accuracy"
    - Venue: SiliconANGLE
    - Date: November 2024
    - URL: https://siliconangle.com/2024/11/20/google-deepminds-alphaqubit-tackles-quantum-error-detection-with-unprecedented-accuracy/

---

## How to Access These Sources

### Priority Order for Reading

**Highest Priority** (Foundational 2024 Breakthroughs):
1. Google Willow (Nature paper) - https://arxiv.org/abs/2408.13687
2. AlphaQubit (Nature) - https://www.nature.com/articles/s41586-024-08148-8
3. Harvard/MIT (Nature 2023) - https://www.nature.com/articles/s41586-023-06927-3

**High Priority** (Recent Implementations):
4. IBM Gross Code (Nature 2024) - https://www.ibm.com/quantum/blog/nature-qldpc-error-correction
5. Riverlane Local Clustering Decoder (Nature Comms 2024)
6. Quantinuum/Microsoft (April 2024) - https://ionq.com/blog/our-novel-efficient-approach

**Important** (ML Benchmarking and RL):
7. Benchmarking ML Models - https://arxiv.org/abs/2311.11167
8. RL Code Optimization - https://quantum-journal.org/papers/q-2019-12-16-215/
9. Real-Time Control RL - https://arxiv.org/abs/2511.08493

**Reference** (Industry Reports and Analysis):
10. Riverlane QEC Report 2024 - https://www.riverlane.com/quantum-error-correction-report-2024
11. IEEE Spectrum - https://spectrum.ieee.org/quantum-error-correction
12. Physics World Breakthrough - https://physicsworld.com/a/two-advances-in-quantum-error-correction-share-the-physics-world-2024-breakthrough-of-the-year/

### Accessing Paywalled Content

- **Nature/npj articles**: Available through institutional access or arXiv preprints
- **IEEE Spectrum**: Free reading available
- **Riverlane Reports**: Free download from website
- **arXiv papers**: Always freely available at arxiv.org
- **Company blogs**: Direct access from Google, IBM, Riverlane, IonQ, etc.

---

**Total Sources Compiled**: 62 primary references
**Coverage**: Comprehensive spanning hardware, ML, RL, theory, and implementation
**Last Updated**: December 28, 2025
**Recommended Update Frequency**: Quarterly (given 120+ new papers published in 2024-2025)

