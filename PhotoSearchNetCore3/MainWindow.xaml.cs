using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;

using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using Microsoft.ML.ImageAnalytics;



namespace PhotoSearchNetCore3
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        public class ImageNetData
        {
            public string Filename;

        }


        public struct ImageNetSettings
        {
            public const int imageHeight = 32;
            public const int imageWidth = 32;
            public const bool channelsLast = true;
            public const float scalevalue = (float)(1.0 / 255);

        }

        private List<ImageNetData> ImageDataset()
        {

            List<ImageNetData> list = new List<ImageNetData>();


            foreach (var filename in Directory.GetFiles(FolderLocation.Text).ToList<string>())
            {
                var filenameonly = System.IO.Path.GetFileName(filename);

                var data = new ImageNetData();
                data.Filename = filenameonly;

                list.Add(data);

            }

            return list;

        }


        public MainWindow()
        {
            InitializeComponent();
            string[] labellist = new string[] { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
            foreach (var label in labellist)
            {
                Keyword.Items.Add(label);                
            }
            Keyword.Items.Add("ALL");
            Keyword.Text = "ALL";
            FolderLocation.Text = @"C:\ML\Photos\";
        }

        private void ImageGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (FileGrid.SelectedIndex > -1)
                ImageViewer1.Source = new BitmapImage(new Uri(FileGrid.SelectedItem.ToString(), UriKind.Absolute));
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e)
        {
            if (Directory.Exists(FolderLocation.Text))
            {
                if (Keyword.SelectedItem.ToString() == "ALL")
                {
                    List<string> files = Directory.GetFiles(FolderLocation.Text).ToList<string>();
                    FileGrid.ItemsSource = files;
                }
                else
                {
                    var modelPath = @"model.onnx";
                    // Inspect the model's inputs and outputs
                    var session = new InferenceSession(modelPath);
                    var inputInfo = session.InputMetadata.First();
                    var outputInfo = session.OutputMetadata.First();

                    Console.WriteLine($"Input Name is {String.Join(",", inputInfo.Key)}");
                    Console.WriteLine($"Input Dimensions are {String.Join(",", inputInfo.Value.Dimensions)}");
                    Console.WriteLine($"Output Name is {String.Join(",", outputInfo.Key)}");
                    Console.WriteLine($"Output Dimensions are {String.Join(",", outputInfo.Value.Dimensions)}");

                    var mlContext = new MLContext();
                    var imagedata = ImageDataset();

                    var trainData = mlContext.CreateStreamingDataView(imagedata);

                    string imagesFolder = FolderLocation.Text;


                    var pipeline = mlContext.Transforms.LoadImages(imageFolder: imagesFolder, columns: ("Filename", "ImageReal"))
                                .Append(mlContext.Transforms.Resize("ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
                                .Append(mlContext.Transforms.ExtractPixels(new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", inputInfo.Key,
                                interleave: ImageNetSettings.channelsLast,scale: ImageNetSettings.scalevalue )}))
                                .Append(new OnnxScoringEstimator(mlContext, modelPath, new[] { inputInfo.Key }, new[] { outputInfo.Key }));


                    // Run the pipeline and get the transformed values
                    var transformedValues = pipeline.Fit(trainData).Transform(trainData);
                    var predictions = transformedValues.GetColumn<float[]>(mlContext, outputInfo.Key).ToList();
                    List<string> resultimage = new List<string>();
                    int counter = 0;

                    foreach (var prediction in predictions)
                    {
                        var predictiondic = prediction.Select((value, index) => new { value, index })
                          .ToDictionary(pair => pair.index, pair => pair.value);
                        var resultlist = prediction.OrderByDescending(s => s).ToList();

                        for (int index = 0; index < 1; index++)
                        {
                            var labelclass = predictiondic.Where(s=>s.Value== resultlist[index]).First().Key;
                            if (labelclass == Keyword.SelectedIndex)
                            {
                                resultimage.Add(imagesFolder + imagedata[counter].Filename);


                            }
                        }
                        counter++;
                    }
                    FileGrid.ItemsSource = resultimage;
                }
            }
            else
            {
                MessageBox.Show("Folder Not Found");
            }
        }
    }
}
