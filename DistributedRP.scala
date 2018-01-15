package Reader

/**
  * Created by ldkj on 17-12-15.
  */
import java.util.Collections

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

import scala.util.Random


object RP {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RP")
    val sc = new SparkContext(conf)

    // libsvm format
    val dataPath = args(0)
    val featureSize = args(4).toInt
    val labelIndexPath = ""
    val labelRatio = 0.2
    val knn = 5
    val minPartitions = args(3).toInt
    // create data graph
    println("=== begin to create graph")
    val graph = createSimilarityGraph(dataPath, featureSize, labelIndexPath, knn, labelRatio, sc, minPartitions=minPartitions)

    //reliability propagation
    val lambda1 = 1.0
    val lambda2 = 1.0
    val maxIter = args(2).toInt
    val step = args(5).toDouble

   println("======= begin to do RP")
   val C = RPropagation(graph, lambda1, lambda2, maxIter, step)

   //save result
   val savePath = args(1)
   println("===== begin to store")
   C.saveAsTextFile(savePath + "_" + lambda1 +"_"+lambda2)


  }

  /**
    * Reliability Propagation
    * return: id, C, R
    * */
  def RPropagation(graph: Graph[(Array[Double], Int, Boolean, Double, Double), Double],
                   lambda1: Double, lambda2: Double, maxIter: Int, step:Double): VertexRDD[(Double, Double)] ={
    var runtime = 0
    var graphTmp = graph
    while (runtime < maxIter){
      /**
        * 1. Update C
        * */
      // message passing,
      // msg = unlabeled info, labeled info, D_i for unlabled
      // msg = Cj*wij, wij, D_i_U
      val proGraph = graphTmp.aggregateMessages[(Double, Double, Double)](
        sendMsg => {
          val wij = sendMsg.attr
          sendMsg.sendToSrc {
            if(sendMsg.dstAttr._3)
              (0.0, -1.0 * wij, 0.0)
            else
              (sendMsg.dstAttr._5 * wij, 0.0, sendMsg.dstAttr._5)
          }
          sendMsg.sendToDst {
            if(sendMsg.srcAttr._3)
              (0.0, -1.0 * wij, 0.0)
            else
              (sendMsg.srcAttr._5 * wij, 0.0, sendMsg.srcAttr._5)
          }
        },
        (msg1, msg2) => {
          (msg1._1 + msg2._1, msg1._2 + msg1._2, msg1._3 + msg2._3)
        }
      )
      //integrate message
      graphTmp = graph.joinVertices(proGraph)((id1, idAttr, newAttr) => {
        val isLabeled = idAttr._3
        if(isLabeled)
          idAttr
        else{
          val ci = idAttr._4
          val ri = idAttr._5
          val gred = newAttr._3 * ci - newAttr._1 + newAttr._2 + lambda2 * (ci - ri)
          val cNew = math.min(1, math.max(0, ci - step * gred))
//                    println(cNew+"==")
          (idAttr._1,idAttr._2,idAttr._3,cNew,idAttr._5)
        }
      })

      /**
        * Update graph
        * */
      graphTmp = graphTmp.mapTriplets(tri => {
        val ci = tri.srcAttr._4
        val cj = tri.dstAttr._4
        val wij = tri.attr
        val wijNew = math.max(0, wij - (ci - cj) / (2 * lambda1))
        wijNew
      })
      println("============= runtime = " + runtime)
      runtime += 1
    }

    val result = graphTmp.vertices.filter(x => !x._2._3).mapValues(attr => (attr._4, attr._5))
    result
  }


  /**
    * format similarity matrix
    * input data format: features, y
    * labelIndexPath: data id, meaning it's labeled (star from 0)
    *
    * node: id, features, label, isLabeled, C, R
    * */
  def createSimilarityGraph(dataPath:String, featureSize:Int, labelIndexPath:String, knn:Int, labelRatio:Double,
                            sc:SparkContext, minPartitions:Int=100): Graph[(Array[Double], Int, Boolean, Double, Double), Double] ={
    val dataRaw = sc.textFile(dataPath, minPartitions=minPartitions)

    var labelIndex:Set[Long] = null
    if(labelIndexPath != null && labelIndexPath.length > 1)
      labelIndex = sc.textFile(labelIndexPath).map(_.trim.toLong).collect().toSet
    else{
      val dataLength = dataRaw.count().toInt
      val in = 0 to (dataLength - 1)
      val indexes = Random.shuffle(in)
      val end = dataLength * labelRatio
      labelIndex = indexes.slice(0, end.toInt).map(_.toLong).toSet
    }

    // id (features, label, isLabeled, C=1.0, R=1.0)
    val data:RDD[(Long, (Array[Double], Int, Boolean, Double, Double))] =
      dataRaw.map(line => {
        val tm = Array.fill[Double](featureSize)(0.0)
        val tmp = line.trim.split(" ")
        val label = tmp(0)
        val features = tmp.slice(1, featureSize).map(x => {val y = x.split(":");(y(0).toInt, y(1).toDouble)})
        features.foreach{case(index, value) => tm(index - 1) = value}
        (tm, label.toInt, 1.0)
      }).zipWithIndex() // id starts from 0
        .map(x => (x._2, (x._1._1, x._1._2, labelIndex.contains(x._2), x._1._3, 1.0))).cache()

    val dataWithoutY = data.map{case(id, (features, label, isLabeled, c, r)) => (id, features)}.cache()
    val WFull = dataWithoutY.cartesian(dataWithoutY)
      .map{case(x1, x2) => (x1._1, x2._1, cal(x1._2, x2._2))}
      .cache()

    val edgeFull = WFull.map{x => Edge(x._1, x._2, x._3)}
    var graphFull = Graph(data, edgeFull).cache()
    val nodesSize = graphFull.numVertices
    val edgesSize = graphFull.numEdges
    println("read " + edgesSize + " edges..." + nodesSize +" nodes and labeled num=" + labelIndex.size +"======")
    //    graphFull.triplets.take(10).foreach(println)
//    data.unpersist()
//    dataWithoutY.unpersist()

    //R calculation
    graphFull = RCalculation(graphFull)
    val mean = math.sqrt((WFull.map(_._3).sum) / (nodesSize * (nodesSize - 1)))

    val knnNeighbour = WFull.map{case(x1, x2, x3) => (x1, (x2, math.exp(-1 * x3 * x3 / (mean * featureSize))))}
      .groupByKey()
      .mapValues(_.toList.sortWith(_._2 > _._2).slice(0, knn))
      .map(x => {
      val id1 = x._1
      x._2.map(y => (id1 + "-" + y._1, y._2))
    }).flatMap(x => x).collectAsMap()

//    WFull.unpersist()

    //Symmetrical knn graph
    val graph = graphFull
      .subgraph(x=> {
        val id1 = x.srcId
        val id2 = x.dstId
        if(id1 == id2)
          false
        else
          knnNeighbour.contains(id1 +"-"+id2) || knnNeighbour.contains(id2 +"-"+id1)
      }).mapEdges(edge => {
      val id1 = edge.srcId
      val id2 = edge.dstId
      val newAttr = math.max(knnNeighbour.getOrElse(id1 +"-"+id2, 0.0), knnNeighbour.getOrElse(id2 +"-"+id1, 0.0))
      newAttr
    }).cache()

//    graphFull.unpersist()
    //get stat.
    val nodeNum = graph.vertices.count()
    val edgeNum = graph.edges.count()
    val nodeLabeledNum = graph.vertices.filter(node => node._2._3).count()
    println("read knn " + edgeNum + " edges..." + nodeNum +" nodes and labeled num=" + nodeLabeledNum +"======")
    //    graph.triplets.foreach(x => println(x.srcId +"=="+x.dstId))
    graph
  }

  /**
    * gaussian similarity
    * */
  def cal(x:Array[Double], y:Array[Double]): Double = math.sqrt(x.zip(y).map(z => (z._1 - z._2)*(z._1 - z._2)).sum)


  /**
    * Reliability Prior Measure
    * */
  def RCalculation(graph: Graph[(Array[Double], Int, Boolean, Double, Double), Double]): Graph[(Array[Double], Int, Boolean, Double, Double), Double] ={
    //calculate R on full graph
    //Msg = Map(label -> wij), only from labeled to unlabel
    val proGraph = graph.aggregateMessages[Map[Int, List[Double]]](
      sendMsg => {
        val wij = sendMsg.attr
        val dstIsLabel = sendMsg.dstAttr._3
        val srcIsLabel = sendMsg.srcAttr._3

        if(dstIsLabel && !srcIsLabel) {
          sendMsg.sendToSrc {
            val dstLabel = sendMsg.dstAttr._2
            Map(dstLabel -> List(wij))
          }
        }
        else if(!dstIsLabel && srcIsLabel){
          sendMsg.sendToDst{
            val srcLabel = sendMsg.srcAttr._2
            Map(srcLabel -> List(wij))
          }
        }
      },
      (msg1, msg2) => {
        (msg1 ++ msg2).map {
          case (key, value) => {
            val vv = value.toBuffer
            vv.append(msg1.getOrElse(key, List[Double]()):_*)
            key -> vv.toList
          }
        }
      }
    )
    println("====" + proGraph.count())
    proGraph.foreach(x=> println(x._1+ "=-="+ x._2.mkString(";")))
//    println("---")
    val R = proGraph.map(x => {
      //find max and second max
      val id = x._1
      val msgs = x._2.map(msg => {
        val label = msg._1
        val min = msg._2.min
        (label, min)
      })
      val min = Array(Double.MaxValue, Double.MaxValue) // min, minsecond
      msgs.values.foreach(msg =>
        if(msg < min(0)){
          min(1) = min(0)
          min(0) = msg
        }
        else if(msg < min(1)){
          min(1) = msg
        }
      )
//      println(x._1 +" "+min(0) + " " + min(0))
      if(min(1) <= 0.0)
        (x._1, 0.0)
      else
        (x._1, (min(1) - min(0))/ min(1))
    })

//    R.saveAsTextFile("hdfs://hadoop10:8020/ddR")

    graph.joinVertices(R)((id1, idAttr, newAttr) => {
      val isLabeled = idAttr._3
      if(!isLabeled)
        (idAttr._1, idAttr._2, idAttr._3,idAttr._4, newAttr)
      else
        (idAttr._1, idAttr._2, idAttr._3,idAttr._4,idAttr._5)
    })
  }
}
