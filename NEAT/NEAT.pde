void setup()
{
  size(1000, 1000, P2D);
  train();
}

NetworkVisualizer nv;

void draw()
{
  background(255);
  pushMatrix();
  translate(width/2, height/2);
  nv.Draw();
  popMatrix();
}

void train()
{
  Genome start = new Genome(2, 1, 5);
  Population p = new Population(150, start, 3f, 1f, 1f, 0.4f, null);
  Network currentnet = null;
  InputTarget[] it =
  {
    new InputTarget(new float[] {1, 0}, 1), 
    new InputTarget(new float[] {0, 1}, 1), 
    new InputTarget(new float[] {1, 1}, 0), 
    new InputTarget(new float[] {0, 0}, 0),
  };
  
  boolean solutionfound = false;
  
  do
  {
    for (int j = 0; j < p.GetPopulationsize(); j++)
    {
      Genome currentgenome = p.NextGenome();
      currentnet = new Network(currentgenome);
      float fitness = 0f;
      for (InputTarget currentit : it)
      {
        fitness += abs(currentit.target - currentnet.FeedForward(currentit.input)[0]);
      }
      fitness = 4 - fitness;
      p.SetFitness(currentgenome, pow(fitness, 2));
      
      if(fitness > 3.6f)
      {
        solutionfound = true;
      }
    }
    
    p.NextGeneration(this);
  } while (!solutionfound);
  
  println("done in", p.GetCurrentgeneration(), "generations");
  
  Genome winnergenome = p.FittestGenome.get(p.GetCurrentgeneration()-1).Genome;
  currentnet = new Network(winnergenome);
  
  for (InputTarget currentit : it)
  {
    println("inputs:", currentit.input[0], currentit.input[1], "expected result", currentit.target, currentnet.FeedForward(currentit.input)[0]);
  }
  
  nv = new NetworkVisualizer(winnergenome, (int) (width * 0.8),  (int) (height * 0.8));
}

class InputTarget
{
  public float[] input;

  public float target;

  public InputTarget(float[] input, float target)
  {
    this.input = input;

    this.target = target;
  }
}
