workspace "mitssac-pausa" "Valuing La Pausa: Quantifying Optimal Pass Timing Beyond Speed using OBSO" {

    model {
        researcher = person "Researcher" "Sports analytics researcher running OBSO and PAUSA computations"

        pausa = softwareSystem "mitssac-pausa" "Computes OBSO (Off-Ball Scoring Opportunity) and PAUSA metrics to value optimal pass timing in football" {
            cli = container "calculate_obso.py" "Main CLI entry point for computing OBSO at event, tracking, or virtual trajectory level" "Python, argparse"
            obsoEngine = container "obso.py" "Core OBSO computation: pitch control * transition * scoring probability" "Python, NumPy"
            pitchControl = container "pitch_control.py" "Pitch control surface model based on Spearman 2018" "Python, NumPy, Numba"
            xThreat = container "xthreat.py" "Expected Threat (xT) framework for valuing ball-progressing actions" "Python, NumPy, scikit-learn"
            loader = container "loader.py" "ElasticLoader: loads and converts tracking/event data from ELASTIC format" "Python, Pandas"
            visualization = container "matplotsoccer.py / trace_snapshot.py" "Soccer pitch plotting and OBSO snapshot visualization" "Python, Matplotlib"
            notebooks = container "Jupyter Notebooks" "Interactive analysis: OBSO visualization, PAUSA computation, player/team rankings" "Jupyter, Python"
            staticData = container "Static Data" "Pre-computed EPV grid, Transition matrix, xT grid" "CSV, JSON" "Database"
        }

        elasticFramework = softwareSystem "ELASTIC Framework" "Event-tracking data synchronization into SPADL format" "External"
        dflDataset = softwareSystem "DFL Public Dataset" "Spatiotemporal match event and tracking data from Bundesliga" "External"

        researcher -> cli "Runs OBSO/PAUSA computations" "CLI"
        researcher -> notebooks "Explores results and performs analysis" "Jupyter"

        cli -> loader "Loads match data via"
        cli -> obsoEngine "Computes OBSO via"
        cli -> pitchControl "Generates pitch control surfaces via"

        obsoEngine -> pitchControl "Uses pitch control from"
        obsoEngine -> staticData "Reads EPV and Transition matrices from"

        loader -> elasticFramework "Reads ELASTIC-format parquet data produced by"
        elasticFramework -> dflDataset "Synchronizes and converts raw data from"

        notebooks -> obsoEngine "Calls OBSO computation"
        notebooks -> xThreat "Computes xT ratings"
        notebooks -> visualization "Renders pitch diagrams via"
        notebooks -> staticData "Loads static models from"

        visualization -> pitchControl "Can overlay pitch control on"
    }

    views {
        systemContext pausa "SystemContext" {
            include *
            autoLayout
        }

        container pausa "Containers" {
            include *
            autoLayout
        }

        styles {
            element "Person" {
                shape Person
                background #08427B
                color #ffffff
            }
            element "Software System" {
                background #1168BD
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438DD5
                color #ffffff
            }
            element "Database" {
                shape Cylinder
            }
            element "Component" {
                background #85BBF0
                color #000000
            }
        }
    }

}
