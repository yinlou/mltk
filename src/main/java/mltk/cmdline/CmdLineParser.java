package mltk.cmdline;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class for command line parser.
 * 
 * @author Yin Lou
 * 
 */
public class CmdLineParser {

	private String name;
	private Object obj;

	private List<Argument> argList;
	private List<Field> fieldList;

	/**
	 * Constructor.
	 * 
	 * @param obj the object.
	 */
	public CmdLineParser(Class<?> clazz, Object obj) {
		this.name = clazz.getCanonicalName();
		this.obj = obj;

		argList = new ArrayList<>();
		fieldList = new ArrayList<>();
		
		processFields(obj.getClass().getFields());
		processFields(obj.getClass().getDeclaredFields());
	}

	/**
	 * Parses the command line arguments.
	 * 
	 * @param args the command line arguments.
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public void parse(String[] args) throws IllegalArgumentException, IllegalAccessException {
		if (args.length % 2 != 0) {
			throw new IllegalArgumentException();
		}
		Map<String, String> map = new HashMap<>();
		for (int i = 0; i < args.length; i += 2) {
			map.put(args[i], args[i + 1]);
		}
		for (int i = 0; i < argList.size(); i++) {
			Field field = fieldList.get(i);
			Argument arg = argList.get(i);
			String value = map.get(arg.name());
			if (value != null) {
				Class<? extends Object> fclass = field.getType();
				field.setAccessible(true);
				if (fclass == String.class) {
					field.set(obj, value);
				} else if (fclass == int.class) {
					field.setInt(obj, Integer.parseInt(value));
				} else if (fclass == double.class) {
					field.setDouble(obj, Double.parseDouble(value));
				} else if (fclass == float.class) {
					field.setFloat(obj, Float.parseFloat(value));
				} else if (fclass == boolean.class) {
					field.setBoolean(obj, Boolean.parseBoolean(value));
				} else if (fclass == long.class) {
					field.setLong(obj, Long.parseLong(value));
				} else if (fclass == char.class) {
					field.setChar(obj, value.charAt(0));
				}
			} else if (arg.required()) {
				throw new IllegalArgumentException();
			}
		}
	}

	/**
	 * Prints the generated usage.
	 */
	public void printUsage() {
		StringBuilder sb = new StringBuilder();
		sb.append("Usage: ").append(name).append("\n");

		StringBuilder required = new StringBuilder();
		StringBuilder optional = new StringBuilder();

		for (Argument arg : argList) {
			if (arg.required()) {
				required.append(arg.name()).append("\t").append(arg.description()).append("\n");
			} else {
				optional.append("[").append(arg.name()).append("]\t").append(arg.description()).append("\n");
			}
		}

		sb.append(required).append(optional);
		System.err.println(sb.toString());
	}
	
	private void processFields(Field[] fields) {
		for (Field field : fields) {
			Argument argument = field.getAnnotation(Argument.class);
			if (argument != null) {
				fieldList.add(field);
				argList.add(argument);
			}
		}
	}

}
