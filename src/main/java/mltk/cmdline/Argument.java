package mltk.cmdline;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/**
 * Command line argument.
 * 
 * @author Yin Lou
 * 
 */
@Retention(java.lang.annotation.RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Argument {

	/**
	 * Name of this argument.
	 * 
	 * @return the name of this argument.
	 */
	String name() default "";

	/**
	 * Description of this argument.
	 * 
	 * @return the description of this argument.
	 */
	String description() default "";

	/**
	 * Whether this argument is required.
	 * 
	 * @return <code>true</code> if the argument is required.
	 */
	boolean required() default false;

}
