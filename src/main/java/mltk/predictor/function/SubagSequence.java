package mltk.predictor.function;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import mltk.util.Element;
import mltk.util.Permutation;
import mltk.util.Queue;
import mltk.util.UFSets;
import mltk.util.tuple.IntPair;

class SubagSequence {
	
	static class SampleDelta {
		
		int[] toAdd;
		int[] toDel;
		
		SampleDelta(Set<Integer> toAdd, Set<Integer> toDel) {
			this.toAdd = new int[toAdd.size()];
			int k = 0;
			for (Integer idx : toAdd) {
				this.toAdd[k++] = idx;
			}
			k = 0;
			this.toDel = new int[toDel.size()];
			for (Integer idx : toDel) {
				this.toDel[k++] = idx;
			}
		}
		
		int getDistance() {
			return toAdd.length + toDel.length;
		}
		
	}
	
	static class Sample {
		
		Set<Integer> set;
		int[] indices;
		
		Sample(Set<Integer> set) {
			this.set = set;
			this.indices = new int[set.size()];
			int k = 0;
			for (Integer idx : set) {
				this.indices[k++] = idx;
			}
		}
		
		static int computeDistance(Sample s1, Sample s2) {
			SampleDelta delta = computeDelta(s1, s2);
			return delta.getDistance();
		}
		
		static SampleDelta computeDelta(Sample s1, Sample s2) {
			Set<Integer> toAdd = new HashSet<>(s2.set);
			toAdd.removeAll(s1.set);
			Set<Integer> toDel = new HashSet<>(s1.set);
			toDel.removeAll(s2.set);
			
			return new SampleDelta(toAdd, toDel);
		}
		
		int getWeight() {
			return set.size();
		}
		
	}
	
	Sample[] samples;
	int[] start;
	int[] end;
	SampleDelta[] deltas;
	int[] count;

	SubagSequence(int n, int m, int baggingIters) {
		samples = new Sample[baggingIters];
		count = new int[baggingIters];
		for (int i = 0; i < samples.length; i++) {
			samples[i] = createSubsample(n, m);
		}
		computeSequence(samples);
	}
	
	Sample createSubsample(int n, int m) {
		Permutation perm = new Permutation(n);
		perm.permute();
		int[] a = perm.getPermutation();
		Set<Integer> set = new HashSet<>(m);
		for (int i = 0; i < m; i++) {
			set.add(a[i]);
		}
		
		return new Sample(set);
	}
	
	private void computeSequence(Sample[] samples) {
		PriorityQueue<Element<IntPair>> q = new PriorityQueue<>(samples.length * (samples.length - 1) / 2);
		for (int i = 0; i < samples.length - 1; i++) {
			for (int j = i + 1; j < samples.length; j++) {
				int distance = Sample.computeDistance(samples[i], samples[j]);
				q.add(new Element<IntPair>(new IntPair(i, j), distance));
			}
		}
		
		Map<Integer, Set<Integer>> map = new HashMap<>();
		
		UFSets ufsets = new UFSets(samples.length);
		while (!q.isEmpty()) {
			Element<IntPair> e = q.poll();
			int x = e.element.v1;
			int y = e.element.v2;
			int root1 = ufsets.find(x);
			int root2 = ufsets.find(y);
			if (root1 != root2) {	
				ufsets.union(root1, root2);
				if (!map.containsKey(x)) {
					map.put(x, new HashSet<>());
				}
				map.get(x).add(y);
				if (!map.containsKey(y)) {
					map.put(y, new HashSet<>());
				}
				map.get(y).add(x);
			}
		}
		
		
		int s = 0;
		for (int i = 1; i < samples.length; i++) {
			int weight = samples[i].getWeight();
			if (weight < samples[s].getWeight()) {
				s = i;
			}
		}
		
		List<Integer> fromList = new ArrayList<>(samples.length);
		List<Integer> toList = new ArrayList<>(samples.length);
		List<SampleDelta> deltas = new ArrayList<>(samples.length);
		
		// BFS
		Set<Integer> covered = new HashSet<>();
		covered.add(s);
		Queue<Integer> queue = new Queue<>();
		queue.enqueue(s);
		while (covered.size() < samples.length) {
			Integer node = queue.dequeue();
			Set<Integer> children = map.get(node);
			for (Integer child : children) {
				if (covered.contains(child)) {
					continue;
				}
				covered.add(child);
				fromList.add(node);
				toList.add(child);
				SampleDelta delta = Sample.computeDelta(samples[node], samples[child]);
				deltas.add(delta);
				queue.enqueue(child);
			}
		}
		
		this.start = new int[fromList.size()];
		this.end = new int[toList.size()];
		this.deltas = new SampleDelta[deltas.size()];
		for (int i = 0; i < this.start.length; i++) {
			this.start[i] = fromList.get(i);
			this.end[i] = toList.get(i);
			this.deltas[i] = deltas.get(i);
		}
		for (int i = 0; i < this.start.length; i++) {
			count[this.start[i]]++;
		}
	}
	
}
