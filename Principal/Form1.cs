using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Encog.Engine.Network.Activation;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation;
using Encog.Neural.Networks.Training.Propagation.Back;

namespace Principal
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double[][] x = { new double[] {0.1, 0.2},
                                 new double[] {0.4, 0.3}
                };
            double[][] y = { new double[] {0.3},
                                 new double[] {0.4}
                };

            BasicNetwork rede = new BasicNetwork();
            rede.AddLayer(new BasicLayer(2));
            rede.AddLayer(new BasicLayer(2));
            rede.AddLayer(new BasicLayer(1));
            rede.Structure.FinalizeStructure();
            rede.Reset();
            BasicMLDataSet dataset = new BasicMLDataSet(x, y);

            Backpropagation propagation = new Backpropagation(rede, dataset, 0.3, 0.7);
            int epoch = 0;
            while (true)
            {
                propagation.Iteration();
                richTextBox1.AppendText("Época " + epoch.ToString() + " Erro " + propagation.Error + Environment.NewLine);
                epoch++;

                if (epoch > 3500 || propagation.Error < 0.01)
                    break;
            };
            foreach (IMLDataPair d in dataset)
            {
                IMLData o = rede.Compute(d.Input);
                richTextBox1.AppendText(" Saída " + o + " Ideal " + d.Input[0] + Environment.NewLine);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            double[][] x = { new double[] {0.1, 0.2},
                             new double[] {0.4, 0.3}
            };
            double[][] y = { new double[] {0.3},
                             new double[] {0.7}
            };
            BasicMLDataSet dataset = new BasicMLDataSet(x, y);

            BasicNetwork rede = new BasicNetwork();
            rede.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            rede.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            rede.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
            rede.Structure.FinalizeStructure();
            rede.Reset();

            Backpropagation train = new Backpropagation(rede, dataset, 0.5, 0.1);

            int epoch = 0;
            do
            {
                train.Iteration();
                if (epoch % 100 == 0)
                    richTextBox2.AppendText("Época " + epoch.ToString() + " Erro " + train.Error + Environment.NewLine);
                epoch++;

            } while (epoch < 3000);

            for (double t = 0.0; t <= 5; t += 0.05)
            {
                double[] d = new double[] { t, t };
                IMLData input = new BasicMLData(d);
                IMLData output = rede.Compute(input);
                double[] result = new double[output.Count];
                output.CopyTo(result, 0, output.Count);
                richTextBox2.AppendText(" " + t + "+" + t + "=" + result[0] + Environment.NewLine);
            }

        }
    }
}
