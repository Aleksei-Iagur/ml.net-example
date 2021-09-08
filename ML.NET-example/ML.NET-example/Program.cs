using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML.NET_example
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView data = mlContext.Data.LoadFromTextFile<Review>("test.tsv");

            var pipeline = mlContext.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(Review.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

            ITransformer model = pipeline.Fit(data);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<Review, Prediction>(model);
            var testReview = new Review { Text = "Всё было отлично!" };
            Print(predictionEngine.Predict(testReview));
            Console.ReadLine();
        }

        private static void Print(Prediction predict)
        {
            Console.WriteLine($"{nameof(predict.Probability)}: {predict.Probability}");
            Console.WriteLine($"{nameof(predict.Score)}: {predict.Score}");
            Console.WriteLine($"{nameof(predict.IsGood)}: {predict.IsGood}");
        }
    }

    public class Review
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1), ColumnName("Label")]
        public bool IsGood { get; set; }
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}