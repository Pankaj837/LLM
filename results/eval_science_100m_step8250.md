# Scientific evaluation: ckpt_best.pt (step 8250)

- question set: 3544 items, manifest matches
- excluded for train-data overlap: 75
- overall acc_norm 26.9% (chance 24.8%, 95% CI 25.5%–28.4%)
- time 386 s on mps

Scores are acc_norm for multiple choice (log-likelihood per character) and exact-within-tolerance for numeric answers. 'above chance' means the lower end of the 95% interval is above the chance level.

## By domain

| domain | n | accuracy | 95% CI | chance | above chance |
|---|---|---|---|---|---|
| astro | 150 | 32.7% | 25.7%–40.5% | 25.0% | yes |
| bio | 290 | 25.5% | 20.8%–30.8% | 25.0% | no |
| cad | 1 | 0.0% | 0.0%–79.3% | 25.0% | no |
| chemistry | 252 | 25.8% | 20.8%–31.5% | 24.5% | no |
| code | 202 | 24.8% | 19.3%–31.1% | 25.0% | no |
| earth | 274 | 32.1% | 26.9%–37.9% | 25.0% | yes |
| electrical_eng | 143 | 32.2% | 25.1%–40.2% | 25.0% | yes |
| engineering | 40 | 12.5% | 5.5%–26.1% | 18.8% | no |
| exp_val | 24 | 20.8% | 9.2%–40.5% | 19.8% | no |
| finance | 358 | 24.6% | 20.4%–29.3% | 25.0% | no |
| math | 345 | 22.3% | 18.2%–27.0% | 25.0% | no |
| multistep | 200 | 1.5% | 0.5%–4.3% | 0.0% | yes |
| physics | 409 | 25.9% | 21.9%–30.4% | 24.7% | no |
| scidisc | 200 | 50.0% | 43.1%–56.9% | 50.0% | no |
| sciml | 16 | 25.0% | 10.2%–49.5% | 25.0% | no |
| sciprog | 4 | 25.0% | 4.6%–69.9% | 25.0% | no |
| sciqa | 599 | 30.4% | 26.8%–34.2% | 25.0% | yes |
| simulation | 22 | 31.8% | 16.4%–52.7% | 25.0% | no |
| tech_comm | 15 | 33.3% | 15.2%–58.3% | 25.0% | no |

## By domain and level

| domain | basic | intermediate | advanced |
|---|---|---|---|
| astro | – | 33% (n=150, chance 25%) | – |
| bio | – | 30% (n=150, chance 25%) | 21% (n=140, chance 25%) |
| cad | 0% (n=1, chance 25%) | – | – |
| chemistry | 0% (n=3, chance 0%) | 28% (n=149, chance 25%) | 24% (n=100, chance 25%) |
| code | 0% (n=3, chance 25%) | 27% (n=103, chance 25%) | 23% (n=96, chance 25%) |
| earth | 41% (n=150, chance 25%) | 21% (n=124, chance 25%) | – |
| electrical_eng | – | 32% (n=143, chance 25%) | – |
| engineering | 0% (n=19, chance 16%) | 25% (n=16, chance 20%) | 20% (n=5, chance 25%) |
| exp_val | 20% (n=10, chance 18%) | 18% (n=11, chance 20%) | 33% (n=3, chance 25%) |
| finance | – | 27% (n=149, chance 25%) | 23% (n=209, chance 25%) |
| math | 17% (n=149, chance 25%) | 22% (n=101, chance 25%) | 32% (n=95, chance 25%) |
| multistep | – | 2% (n=200, chance 0%) | – |
| physics | 23% (n=158, chance 25%) | 31% (n=154, chance 25%) | 24% (n=97, chance 25%) |
| scidisc | – | – | 50% (n=200, chance 50%) |
| sciml | 67% (n=3, chance 25%) | 25% (n=8, chance 25%) | 0% (n=5, chance 25%) |
| sciprog | 25% (n=4, chance 25%) | – | – |
| sciqa | 37% (n=300, chance 25%) | 24% (n=299, chance 25%) | – |
| simulation | 43% (n=7, chance 25%) | 33% (n=9, chance 25%) | 17% (n=6, chance 25%) |
| tech_comm | 25% (n=8, chance 25%) | 43% (n=7, chance 25%) | – |

## By question type and skill

| group | n | accuracy | chance |
|---|---|---|---|
| mcq | 3319 | 28.7% | 26.5% |
| numeric | 225 | 1.8% | 0.0% |
| code_reading | 8 | 12.5% | 25.0% |
| concept | 56 | 26.8% | 25.0% |
| design | 6 | 16.7% | 25.0% |
| formatting | 2 | 50.0% | 25.0% |
| formula_selection | 47 | 8.5% | 16.0% |
| insufficient_info | 7 | 42.9% | 25.0% |
| multi_step | 9 | 11.1% | 11.1% |
| unit_conversion | 9 | 22.2% | 16.7% |
| unsupported_claim | 1 | 100.0% | 25.0% |
| validation | 4 | 50.0% | 25.0% |

## Error types (wrong answers only)

For hand-written multiple-choice questions every wrong option is labelled (wrong formula, unit error, calculation error, misconception, unsupported claim, ...), so a wrong pick shows which mistake the model prefers.

| error | count |
|---|---|
| wrong_choice | 2274 |
| no_number | 137 |
| wrong_value | 79 |
| misconception | 62 |
| wrong_formula | 16 |
| calculation_error | 6 |
| unit_error | 5 |
| unsupported_claim | 5 |
| repetition_loop | 3 |
| unit_error (off by 10^-3) | 1 |
| unit_error (off by 10^-1) | 1 |

## Examples (real model behaviour, first items of each domain)

**astro / intermediate / correct** `mmlu-astronomy/test/72`  
Q: What effect or effects would be most significant if the Moon's orbital plane were exactly the same as the ecliptic plane?  
Model chose: Solar eclipses would be much more frequent.  
Reference: Solar eclipses would be much more frequent. (error type: correct)

**astro / intermediate / correct** `mmlu-astronomy/test/111`  
Q: The axis of the Earth is tilted at an angle of approximately ... relative to the orbital plane around the Sun.  
Model chose: 23.5 degrees  
Reference: 23.5 degrees (error type: correct)

**astro / intermediate / wrong** `mmlu-astronomy/test/64`  
Q: Which of the following characteristics would not necessarily suggest that a rock we found is a meteorite.  
Model chose: It has different elemental composition than earth  
Reference: It is highly processed (error type: wrong_choice)

**astro / intermediate / wrong** `mmlu-astronomy/test/104`  
Q: 20000 years from now ...  
Model chose: The Moon will be closer to the Earth and the Earth’s day will be shorter.  
Reference: The Moon will be further from the Earth and the Earth’s day will be longer. (error type: wrong_choice)

**bio / intermediate / correct** `mmlu-high_school_biology/test/204`  
Q: A small population of birds is blown to an isolated island by a hurricane and establishes a colony. The gene pool of the newly established population of birds is likely to reflect the effects of which of the following?  
Model chose: Genetic drift  
Reference: Genetic drift (error type: correct)

**bio / intermediate / correct** `mmlu-high_school_biology/test/101`  
Q: During drought years on the Galapagos, small, easily eaten seeds become rare, leaving mostly large, hard-cased seeds that only birds with large beaks can eat. If a drought persists for several years, what should one expect to result from natural selection?  
Model chose: More small-beaked birds dying than larger-beaked birds. The offspring produced in subsequent generations have a higher percentage of birds with large beaks.  
Reference: More small-beaked birds dying than larger-beaked birds. The offspring produced in subsequent generations have a higher percentage of birds with large beaks. (error type: correct)

**bio / intermediate / wrong** `mmlu-high_school_biology/test/286`  
Q: If a forest of fir, birch, and white spruce trees was devastated by fire, which of the following would most likely happen?  
Model chose: The number of species would stabilize as the ecosystem matures.  
Reference: Secondary succession would begin to occur. (error type: wrong_choice)

**bio / intermediate / wrong** `mmlu-high_school_biology/test/208`  
Q: African lungfish often live in small, stagnant, freshwater pools. With reduced gills, they differ from other fish in that they breathe air. They also differ in that they release the nitrogen waste urea instead of ammonia. Which of the following statements correctly explains how this is an advantage   
Model chose: Urea is insoluble in water and sinks to the bottom of the small pools. Thus the poisonous substance is isolated from the fish.  
Reference: Ammonia, which is highly toxic, would endanger lungfish that inhabit the small pools. (error type: wrong_choice)

**cad / basic / wrong** `hw-cad-01`  
Q: In a CAD program, the extrude operation  
Model chose: spins a 2D sketch around an axis to make a solid of revolution  
Reference: pushes a 2D sketch along a direction to make a 3D solid (error type: misconception)

**chemistry / intermediate / correct** `mmlu-high_school_chemistry/test/86`  
Q: In determining the order for reactant A in a certain reaction, the concentrations of all other reactants are held constant while the concentration of A is tripled from one experiment to another. The reaction rate is found to triple. The appropriate exponent for A in the rate law is  
Model chose: 1  
Reference: 1 (error type: correct)

**chemistry / intermediate / correct** `mmlu-high_school_chemistry/test/113`  
Q: Changing which of the following will change the numerical value of the equilibrium constant?  
Model chose: The temperature  
Reference: The temperature (error type: correct)

**chemistry / basic / wrong** `hw-chn-01`  
Q: How many moles of water are in 36 g of water (molar mass 18 g/mol)?  
Model wrote: `6` → parsed 6.0  
Reference: 2 mol (error type: wrong_value)

**chemistry / basic / wrong** `hw-chn-02`  
Q: What is the pH of a 0.001 M solution of hydrochloric acid (fully dissociated)?  
Model wrote: `0.001 M` → parsed 0.001  
Reference: 3  (error type: wrong_value)

**code / intermediate / correct** `mmlu-high_school_computer_science/test/61`  
Q: Which of the following is a characteristic of the fault-tolerant nature of routing on the Internet?  
Model chose: The ability to provide data transmission even when some connections have failed  
Reference: The ability to provide data transmission even when some connections have failed (error type: correct)

**code / intermediate / correct** `mmlu-high_school_computer_science/test/47`  
Q: The boolean expression a[i] == max || !(max != a[i]) can be simplified to  
Model chose: a[i] == max  
Reference: a[i] == max (error type: correct)

**code / basic / wrong** `hw-co-01`  
Q: In Python, what does 7 // 2 evaluate to?  
Model chose: 1  
Reference: 3 (error type: misconception)

**code / basic / wrong** `hw-co-02`  
Q: In Python, what does 0.1 + 0.2 == 0.3 evaluate to?  
Model chose: 0.3  
Reference: False (error type: misconception)

**earth / basic / correct** `arc-NCEOGA_2013_5_49`  
Q: What is the cause of rising surface temperatures of the Pacific Ocean, drought in the western United States, and flooding on the Pacific coast of South America?  
Model chose: El Niño  
Reference: El Niño (error type: correct)

**earth / basic / correct** `arc-Mercury_SC_405231`  
Q: A scientist found fossils of ocean shells on a mountain top. Based on this find, which statement can be made?  
Model chose: The mountain was covered by water.  
Reference: The mountain was covered by water. (error type: correct)

**earth / basic / wrong** `arc-ACTAAP_2010_7_17`  
Q: Which is an effect that cities have on the weather?  
Model chose: Large numbers of houses cause the wind to be stronger.  
Reference: Large areas of pavement cause the temperature to be hotter. (error type: wrong_choice)

**earth / basic / wrong** `arc-Mercury_SC_415001`  
Q: When is the most energy transferred from the Sun to water in the water cycle?  
Model chose: during precipitation  
Reference: during evaporation (error type: wrong_choice)

**electrical_eng / intermediate / correct** `mmlu-electrical_engineering/test/102`  
Q: The latching current of SCR is 20 mA. Its holding current will be  
Model chose: 10 mA.  
Reference: 10 mA. (error type: correct)

**electrical_eng / intermediate / correct** `mmlu-electrical_engineering/test/141`  
Q: The feedback factor of a Wien bridge oscillator using Op-Amp is  
Model chose: 1/2  
Reference: 1/2 (error type: correct)

**electrical_eng / intermediate / wrong** `mmlu-electrical_engineering/test/36`  
Q: Magnetic flux has the unit of  
Model chose: Ampere turn  
Reference: Weber (error type: wrong_choice)

**electrical_eng / intermediate / wrong** `mmlu-electrical_engineering/test/134`  
Q: Snubber circuit is used with SCR  
Model chose: either series or parallel.  
Reference: in parallel. (error type: wrong_choice)

**engineering / intermediate / correct** `hw-eng-07`  
Q: Water (kinematic viscosity 1×10⁻⁶ m²/s) flows at 2 m/s through a pipe of 50 mm diameter. The flow is  
Model chose: turbulent, with a Reynolds number of about 100,000  
Reference: turbulent, with a Reynolds number of about 100,000 (error type: correct)

**engineering / intermediate / correct** `hw-eng-10`  
Q: A pump delivers water at a flow rate of 0.01 m³/s. What hydraulic power does it deliver?  
Model chose: It cannot be determined: the head is not given  
Reference: It cannot be determined: the head is not given (error type: correct)

**engineering / basic / wrong** `hw-eng-01`  
Q: A steel rod (E = 200 GPa) is 2 m long, has a cross-section of 100 mm² and carries a tensile force of 10 kN. Its elongation is  
Model chose: 0.1 mm  
Reference: 1 mm (error type: unit_error)

**engineering / intermediate / wrong** `hw-eng-02`  
Q: A simply supported beam of span L carries a single point load P at mid-span. The maximum bending moment is  
Model chose: PL/2  
Reference: PL/4 (error type: wrong_formula)

**exp_val / basic / correct** `hw-ev-11`  
Q: How many significant figures does 0.004560 have?  
Model chose: 4  
Reference: 4 (error type: correct)

**exp_val / advanced / correct** `hw-ev-14`  
Q: A meaningful validation comparison between a simulation and an experiment needs  
Model chose: both the experimental uncertainty and the numerical uncertainty  
Reference: both the experimental uncertainty and the numerical uncertainty (error type: correct)

**exp_val / basic / wrong** `hw-ev-01`  
Q: Sixteen repeated measurements have a sample standard deviation of 2.0. The standard error of the mean is  
Model chose: 2.0  
Reference: 0.5 (error type: misconception)

**exp_val / intermediate / wrong** `hw-ev-02`  
Q: Two independent uncertainty components are 3 units and 4 units. Their combined standard uncertainty is  
Model chose: 1 unit  
Reference: 5 units (error type: wrong_formula)

**finance / intermediate / correct** `mmlu-high_school_macroeconomics/test/306`  
Q: The short-run Phillips curve depicts the ____ relationship between ____ and ____.  
Model chose: negative the inflation rate the unemployment rate  
Reference: negative the inflation rate the unemployment rate (error type: correct)

**finance / intermediate / correct** `mmlu-high_school_macroeconomics/test/36`  
Q: An increase in the price of forklifts imported into the United States from Belgium will  
Model chose: have no effect on the consumer price index or the GDP deflator.  
Reference: have no effect on the consumer price index or the GDP deflator. (error type: correct)

**finance / intermediate / wrong** `mmlu-high_school_macroeconomics/test/175`  
Q: At the break-even point  
Model chose: household spending equals household income.  
Reference: total spending equals income. (error type: wrong_choice)

**finance / intermediate / wrong** `mmlu-high_school_macroeconomics/test/223`  
Q: If a bank has $1000 in checking deposits and the bank is required to reserve $250 what is the reserve ratio? How much does the bank have in excess reserves? What is the size of the money multiplier?  
Model chose: 75% $250 M = 4  
Reference: 25% $750 M = 4 (error type: wrong_choice)

**math / basic / correct** `mmlu-elementary_mathematics/test/37`  
Q: Which expression represents the phrase below? 3 fewer than a number, p  
Model chose: p-3  
Reference: p-3 (error type: correct)

**math / basic / correct** `mmlu-elementary_mathematics/test/154`  
Q: A group of hikers climbed from Salt Flats (elevation −55 feet)to Talon Bluff (elevation 620 feet). What is the difference in elevation between Talon Bluff and Salt Flats?  
Model chose: 675 feet  
Reference: 675 feet (error type: correct)

**math / basic / wrong** `mmlu-elementary_mathematics/test/39`  
Q: Let 12-9+c=12. What value of c makes the equation true?  
Model chose: 12  
Reference: 9 (error type: wrong_choice)

**math / basic / wrong** `mmlu-elementary_mathematics/test/179`  
Q: Use the Associative Property of Multiplication to simplify 8 • (−7) • (−1).  
Model chose: 56  
Reference: −63 (error type: wrong_choice)

**multistep / intermediate / correct** `gsm8k-test/233`  
Q: There are 36 penguins sunbathing in the snow.  One-third of them jump in and swim in the ocean.  Another one-third go inside the cave to eat their dinner.  How many penguins are still left sunbathing?  
Model wrote: `If there are 36 penguins in the snow, then there are 36/3 = 12 penguins left sunbathing.
If there are 36 penguins left sunbathing, then there are 36/3 = 12 penguins left sunbathing.
The answer is 12.` → parsed 12.0  
Reference: 12  (error type: correct)

**multistep / intermediate / correct** `gsm8k-test/842`  
Q: While on vacation in Bali, Thea bought a hat from a craftsman worth $70. If she gave the craftsman four $20 bills, how much change did she get?  
Model wrote: `If Betty gave the craftsman four $20 bills, how much change did she get?

Let's denote the amount of money Betty gave the hat as P.

Betty gave the hat $P - $70.

Since she bought half of the remaini` → parsed 10.0  
Reference: 10  (error type: correct)

**multistep / intermediate / wrong** `gsm8k-test/1215`  
Q: Tim enters a competition and has to try and guess the number of red jelly beans in a jar.  The jar has a square base and is 6 inches by 6 inches and 15 inches tall.  He knows that a jelly bean is .15 cubic inches.  He also knows that they get about 80% efficiency in packing.  He estimates 30% of the  
Model wrote: `Let's assume the number of red jelly beans in the container is x. Then the number of red jelly beans in the container is (x/2) * (x/2) = (x/2) * (x/2) = (x/2) * (x/2) = (x/2) * (x/2) = (x/2) * (x/2)` → parsed None  
Reference: 36  (error type: no_number)

**multistep / intermediate / wrong** `gsm8k-test/139`  
Q: In a candy machine, there are 22 more than four times the number of pink gumballs as there are blue gumballs. If there are 12 blue gumballs how many pink ones are there?  
Model wrote: `In a candy machine, there are 22 more than four times the number of pink gumballs as there are blue gumballs. If there are 12 blue gumballs how many pink ones are there?
Answer: In a candy machine, t` → parsed None  
Reference: 70  (error type: no_number)

**physics / basic / correct** `hw-ph-02`  
Q: Standard atmospheric pressure is about  
Model chose: 101.3 kPa  
Reference: 101.3 kPa (error type: correct)

**physics / intermediate / correct** `hw-ph-06`  
Q: The energy of a photon of wavelength 500 nm is about  
Model chose: 2.48 eV  
Reference: 2.48 eV (error type: correct)

**physics / basic / wrong** `hw-phn-01`  
Q: Convert 72 km/h to metres per second.  
Model wrote: `72` → parsed 72.0  
Reference: 20 m/s (error type: wrong_value)

**physics / intermediate / wrong** `hw-phn-02`  
Q: What is the period, in seconds, of a simple pendulum 0.25 m long (g = 9.81 m/s²)?  
Model wrote: `10.5` → parsed 10.5  
Reference: 1.003 s (error type: wrong_value)

**scidisc / advanced / correct** `pubmedqa-9381529`  
Q: Using murine models, we have shown that the lysosomotropic amine, chloroquine, is effective in the prevention of graft-versus-host disease (GVHD) mediated by donor T cells reactive with recipient minor histocompatibility antigens (MiHCs). Because lysosomotropic amines can suppress major histocompati  
Model chose: yes  
Reference: yes (error type: correct)

**scidisc / advanced / correct** `pubmedqa-27908583`  
Q: To report the outcomes of surgical treatment of lower limb fractures in patients with chronic spinal cord injuries. A total of 37 lower limb fractures were treated from 2003 to 2010, of which 25 fractures were treated surgically and 12 orthopaedically. Patients of the surgical group had better clini  
Model chose: yes  
Reference: yes (error type: correct)

**scidisc / advanced / wrong** `pubmedqa-11334578`  
Q: To evaluate the effectiveness of feeding information on pharmacy back to primary care doctors in order to create awareness (knowledge) of pharmaceutical expenditure (PE). Retrospective cross-sectional study, through personal interview. Reformed PC, Sabadell, Barcelona. The 80 PC doctors working with  
Model chose: yes  
Reference: no (error type: wrong_choice)

**scidisc / advanced / wrong** `pubmedqa-18375490`  
Q: Human chorionic gonadotropin (hCG) is a heterodimeric glycoprotein hormone that exists in multiple forms. Immunoassays commonly used in clinical laboratories measure intact hCG, total beta hCG (intact hCG + hCG free beta-subunit), and/or hCG free beta-subunit. Measurement of serum concentrations of   
Model chose: yes  
Reference: no (error type: wrong_choice)

**sciml / intermediate / correct** `hw-ml-04`  
Q: A key property claimed for Fourier neural operators is that a trained model  
Model chose: can be evaluated on grids of a different resolution than in training  
Reference: can be evaluated on grids of a different resolution than in training (error type: correct)

**sciml / intermediate / correct** `hw-ml-07`  
Q: To infer an unknown diffusion coefficient with a PINN (an inverse problem), one typically  
Model chose: makes the coefficient a trainable parameter, fitted with the network to data and the PDE residual  
Reference: makes the coefficient a trainable parameter, fitted with the network to data and the PDE residual (error type: correct)

**sciml / basic / wrong** `hw-ml-01`  
Q: In a physics-informed neural network (PINN), the PDE residual is usually evaluated  
Model chose: by comparing the network with a lookup table  
Reference: at collocation points, via automatic differentiation of the network (error type: misconception)

**sciml / intermediate / wrong** `hw-ml-02`  
Q: Spectral bias of neural networks means that PINNs tend to  
Model chose: ignore the boundary conditions  
Reference: learn low frequencies first and struggle with high-frequency or multiscale solutions (error type: misconception)

**sciprog / basic / correct** `hw-sp-02`  
Q: Complete the Python function that adds two numbers:
def add(a, b):
    return  
Model chose: a + b  
Reference: a + b (error type: correct)

**sciprog / basic / wrong** `hw-sp-01`  
Q: In Python, what does this line print?
print("Hello, World!")  
Model chose: "Hello, World!"  
Reference: Hello, World! (error type: misconception)

**sciprog / basic / wrong** `hw-sp-03`  
Q: In Python, what is the value of x after this line?
x = 2 + 3  
Model chose: 2 + 3  
Reference: 5 (error type: misconception)

**sciqa / basic / correct** `arc-MCAS_2004_5_8`  
Q: When an electric fan is running, most of the incoming electrical energy changes into which kind of energy?  
Model chose: mechanical energy  
Reference: mechanical energy (error type: correct)

**sciqa / basic / correct** `arc-Mercury_7221463`  
Q: Which material is the best conductor of heat at room temperature?  
Model chose: copper  
Reference: copper (error type: correct)

**sciqa / basic / wrong** `arc-Mercury_7008593`  
Q: What are the building blocks of all matter?  
Model chose: cells  
Reference: atoms (error type: wrong_choice)

**sciqa / basic / wrong** `arc-Mercury_7205328`  
Q: Which of the following results when hydrogen and chlorine chemically bond to form hydrogen chloride?  
Model chose: a precipitate  
Reference: a compound (error type: wrong_choice)

**simulation / basic / correct** `hw-sim-02`  
Q: The heat equation u_t = α u_xx is classified as  
Model chose: parabolic  
Reference: parabolic (error type: correct)

**simulation / basic / correct** `hw-sim-03`  
Q: The wave equation u_tt = c² u_xx is classified as  
Model chose: hyperbolic  
Reference: hyperbolic (error type: correct)

**simulation / intermediate / wrong** `hw-sim-01`  
Q: An explicit upwind scheme for linear advection at speed u = 2 m/s uses a grid spacing Δx = 0.01 m. For a Courant number of at most 1, the largest stable time step is  
Model chose: 0.02 s  
Reference: 0.005 s (error type: wrong_formula)

**simulation / intermediate / wrong** `hw-sim-04`  
Q: The explicit forward-time centred-space (FTCS) scheme for the 1D heat equation is stable when r = αΔt/Δx² satisfies  
Model chose: any r, it is unconditionally stable  
Reference: r ≤ 1/2 (error type: misconception)

**tech_comm / basic / correct** `hw-tc-02`  
Q: Which is the correct SI way to write a mass of five kilograms?  
Model chose: 5 kg  
Reference: 5 kg (error type: correct)

**tech_comm / intermediate / correct** `hw-tc-04`  
Q: In an IETF specification following RFC 2119, the keyword MUST means  
Model chose: an absolute requirement of the specification  
Reference: an absolute requirement of the specification (error type: correct)

**tech_comm / basic / wrong** `hw-tc-01`  
Q: The abstract of a technical paper should  
Model chose: contain only the background of the field and the history of the problem  
Reference: summarise the objective, method, key results and conclusion (error type: misconception)

**tech_comm / basic / wrong** `hw-tc-03`  
Q: A good figure caption in an engineering report should  
Model chose: repeat the title of the report and the section number in which the figure appears  
Reference: say what is shown, under which conditions and in which units (error type: misconception)

