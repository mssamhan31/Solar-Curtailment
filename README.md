# CANVAS-Open-Source

CANVAS (Curtailment and Network Voltage Analysis Study) is a research project led by Dr. Baran Yildiz which is funded by RACE for 2030. 
Because the high penetration number of PV System in Australia, grid voltage is sometimes high, which triggers curtailment effect.
This CANVAS-Open-Source project is a tool to detect curtailment and measure the amount of energy curtailed in three modes:
1. Tripping (the inverter stops operating in the high voltage condition)
2. V-VAr Response (VAr absorbtion and injection of inverter limits the maximum real power in the high voltage condition)
3. V-Watt Response (The inverter limits the maximum real power production in the high voltage condition)

Given the historical time-series data of ghi, voltage, real power, reactive power, and site information like maximum ac capacity of the inverter, this tool aims to give outputs:
a.	Does the solar inverter trip?
b.	How much is the curtailment due to tripping curtailment in kWh/day?
c.	Does the solar inverter show V-VAr response?
d.	How much is the curtailment due to V-VAr response in kWh/day?
e.	Does the solar inverter show V-Watt response?
f.	How much is the curtailment due to V-Watt curtailment in kWh/day?

This tool will benefit anyone who wants to study PV-Curtailment, and the improved understanding of curtailment could lead to higher levels of PV System integration. 

## Some Related Articles and Papers
1. https://greenreview.com.au/energy/rooftop-solar-pv-curtailment-raises-fairness-concerns/
2. https://theconversation.com/solar-curtailment-is-emerging-as-a-new-challenge-to-overcome-as-australia-dashes-for-rooftop-solar-172152
3. https://www.racefor2030.com.au/wp-content/uploads/2021/11/CANVAS-Succinct-Final-Report_11.11.21.pdf

## Prerequisites

This project runs completely in python with common libraries.
The sample historical dataset will be available soon.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Project Partners
The project partners are AGL, SAPN, Solar Analytics, UNSW

## Authors

* **Naomi M Stringer** - *Tripping Algorithm*
* **Baran Yildiz** - *VVAr Algorithm*
* **Tim Klymenko** - *VWatt Algorithm*
* **M. Syahman Samhan** - *Merging, Open Source Implementation, Debugging*

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 
MIT License is chosen due to its simplicity, yet sufficient for a general use open-source tool.

## Acknowledgments
